import logging
import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm.auto import tqdm
import sparse

from nimare.meta.cbma.ale import ALE, SCALE
from nimare.stats import null_to_p
from nimare.transforms import p_to_z
from nimare.utils import (
    _check_ncores,
    mm2vox,
    vox2mm,
    use_memmap
)
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img

import cupy
from numba import cuda
from cupyx.scipy import ndimage as cupy_ndimage
from numpy.lib.histograms import _get_bin_edges

from nimare_gpu.stats import nullhist_to_p 
from nimare_gpu.utils import get_ale_kernel, _calculate_cluster_measures

LGR = logging.getLogger(__name__)

# GPU kernels
@cuda.jit()
def compute_ma_gpu(
        ma_maps, ijks,
        kernels, exp_starts, exp_lens,
        mask_idx_mapping,
        shape0, shape1, shape2,
        mid, mid1):
    """
    Calculate MA maps on GPU by parallelizing across permutations (block.X),
    experiments (block.Y) and foci (thread)

    ma_maps: (cp.ndarray) (n_perm, n_exp, n_voxels_in_mask)
        MA maps of the current batch of permutations for each experiment
    ijks: (cp.ndarray) (n_perm, n_peak, 3)
        ijk coordinates of peaks for each permutation (can be true or random)
    kernels: (cp.ndarray) (n_exp, kernel_size, kernel_size, kernel_size)
        kernels for each experiment. 
        Note that kernel_size must be the same for all experiments. Therefore
        do not clip out zeros from kernels.
    exp_starts: (cp.ndarray) (n_exp,)
        start index of each experiment in ijks
    exp_lens: (cp.ndarray) (n_exp,)
        number of peaks in each experiment
    mask_idx_mapping: (cp.ndarray) (shape0, shape1, shape2)
        mapping from ijk coordinates to voxel index in mask, i.e.,
        mask_xyz_map[i, j, k] = voxel_index_in_mask if voxel is in mask, otherwise -1
    shape0, shape1, shape2: (int)
        shape of template
    mid, mid1: (int)
        mid = kernel_size // 2
        mid1 = kernel_size - mid
        as kernel size is fixed these should be precalculated on cpu
        to avoid repeated calculation in the gpu
    """
    i_perm = cuda.blockIdx.x # index of permutation in ma_maps
    i_exp = cuda.blockIdx.y # index of experiment
    i_peak = cuda.threadIdx.x # index of peak coordinate within current experiment
    
    # only run if i_peak is a valid index
    if i_peak < exp_lens[i_exp]: 
        # get the global index of peak across all experiments
        peak_idx=exp_starts[i_exp]+i_peak 
        # identify valid voxels around the peak within
        # the range of the kernel
        i = ijks[i_perm, peak_idx, 0]
        j = ijks[i_perm, peak_idx, 1]
        k = ijks[i_perm, peak_idx, 2]
        xl = max(i - mid, 0)
        xh = min(i + mid1, shape0)
        yl = max(j - mid, 0)
        yh = min(j + mid1, shape1)
        zl = max(k - mid, 0)
        zh = min(k + mid1, shape2)
        xlk = mid - (i - xl)
        xhk = mid - (i - xh)
        ylk = mid - (j - yl)
        yhk = mid - (j - yh)
        zlk = mid - (k - zl)
        zhk = mid - (k - zh)

        if (
            (xl >= 0)
            & (xh >= 0)
            & (yl >= 0)
            & (yh >= 0)
            & (zl >= 0)
            & (zh >= 0)
            & (xlk >= 0)
            & (xhk >= 0)
            & (ylk >= 0)
            & (yhk >= 0)
            & (zlk >= 0)
            & (zhk >= 0)
        ):
            for xi in range(xh-xl):
                for yi in range(yh-yl):
                    for zi in range(zh-zl):
                        # loop through all voxels in the kernel
                        # and update MA map if voxel is in mask
                        # `vox` is the index of voxel in mask
                        vox = mask_idx_mapping[xl+xi, yl+yi, zl+zi]
                        if vox!=-1: 
                            # using atomic max to prevent
                            # memory race (though is unlikely)
                            cuda.atomic.max(ma_maps, (i_perm, i_exp, vox), kernels[i_exp, xlk+xi, ylk+yi, zlk+zi])

@cuda.jit(cache=False)
def compute_hist(X, bins, n_bins, Y):
    """
    Calculates histogram of X (n_var, n_obs) and stores it in Y (n_var, n_bins).
    The histogram of each row (variable) is calculated independently and in parallel,
    on different blocks (grid.x). Each thread does the calculations of one observation
    and the threads may be distributed across several blocks (grid.y) depending
    on the number of observations and the number of threads per block (block.x).

    Parameters
    ----------
    X: (cp.ndarray) (n_var, n_obs)
        data to calculate histogram of
    bins: (cp.ndarray) (n_bins,)
        bins of histogram
    n_bins: (int)
        number of bins
    Y: (cp.ndarray) (n_var, n_bins)
        histogram of X
    """
    # get current observation value
    row_idx = cuda.blockIdx.x
    element_idx = cuda.threadIdx.x + cuda.blockIdx.y * cuda.blockDim.x
    if (row_idx >= X.shape[0]) or (element_idx >= X.shape[1]):
        return
    x = X[row_idx, element_idx]
    if (x < bins[0] or bins[n_bins-1] < x):
        # to ignore NaNs, set them to a value outside the bins
        # e.g. -1 for ALE nulls
        return
    # find the bin that x belongs to
    high = n_bins - 1
    low = 0
    while (high - low > 1):
        mid = int((high + low) / 2)
        if bins[mid] <= x:
            low = mid
        else:
            high = mid
    # add 1 to the count of the bin
    # (must be done atomically)
    cuda.atomic.add(Y, (row_idx, low), 1)


# GPU adapation of NiMARE's ALE and SCALE
class DeviceMixin:
    """
    Mixin class for GPU-enabled estimators.
    """
    # TODO: add user option to change this, currently only can changed
    # after object is created via .sigma_scale
    sigma_scale = 1.0 
    # set precision to single precision by default
    # this can be changed by calling .set_dtype(bits)
    d_float = cupy.float32
    c_float = np.float32

    def set_dtype(self, bits):
        """
        Set precision of floating point numbers to bits.
        """
        if bits == 32:
            self.d_float = cupy.float32
            self.c_float = np.float32
        elif bits == 64:
            self.d_float = cupy.float64
            self.c_float = np.float64
        else:
            raise ValueError("bits must be 32 or 64.")

    def _set_kernels_gpu(self):
        """
        Precalculates ALE kernels for each experiment and copies them to GPU.
        This method must be called after _collect_data and _preprocess_data
        """
        mask = self.masker.mask_img
        # from ALEKernel._transform
        ijks = self.inputs_["coordinates"][["i", "j", "k"]].values
        exp_idx = self.inputs_["coordinates"]["id"].values
        use_dict = True
        kernel = None
        if self.kernel_transformer.sample_size is not None:
            sample_sizes = self.kernel_transformer.sample_size
            use_dict = False
        elif self.kernel_transformer.fwhm is None:
            sample_sizes = self.inputs_["coordinates"]["sample_size"].values
        else:
            sample_sizes = None
        if self.kernel_transformer.fwhm is not None:
            assert np.isfinite(self.kernel_transformer.fwhm), "FWHM must be finite number"
            _, kernel = get_ale_kernel(mask, fwhm=self.kernel_transformer.fwhm, sigma_scale=self.sigma_scale)
            use_dict = False

        # from utils.compute_ale_ma
        if use_dict:
            if kernel is not None:
                LGR.warn("The kernel provided will be replace by an empty dictionary.")
            kernels = {}  # retain kernels in dictionary to speed things up
            if not isinstance(sample_sizes, np.ndarray):
                raise ValueError("To use a kernel dictionary sample_sizes must be a list.")
        elif sample_sizes is not None:
            if not isinstance(sample_sizes, int):
                raise ValueError("If use_dict is False, sample_sizes provided must be integer.")
        else:
            if kernel is None:
                raise ValueError("3D array of smoothing kernel must be provided.")
        if exp_idx is None:
            exp_idx = np.ones(len(ijks))

        exp_idx_uniq, exp_idx = np.unique(exp_idx, return_inverse=True)

        # calculate and store kernels for each experiment
        # in addition to start index and number of peaks in each experiment
        self.kernels = [] # TODO: can be confused with kernels dictionary
        for i_exp, _ in enumerate(exp_idx_uniq):
            # Index peaks by experiment
            curr_exp_idx = exp_idx == i_exp
            if use_dict:
                # Get sample_size from input
                sample_size = sample_sizes[curr_exp_idx][0]
                if sample_size not in kernels.keys():
                    _, kernel = get_ale_kernel(mask, sample_size=sample_size, sigma_scale=self.sigma_scale)
                    kernels[sample_size] = kernel
                else:
                    kernel = kernels[sample_size]
            elif sample_sizes is not None:
                _, kernel = get_ale_kernel(mask, sample_size=sample_sizes, sigma_scale=self.sigma_scale)
            self.kernels.append(kernel)

        # convert to array and copy to GPU
        self.d_kernels = cupy.asarray(np.array(self.kernels), dtype=self.d_float)

        # set mid index of the kernels
        self.mid = int(np.floor(self.kernels[0].shape[0] / 2.0))

    def _set_exp_indexing_gpu(self):
        """
        Sets start index and number of peaks in each experiment and copies them to GPU.
        This method must be called after _collect_data and _preprocess_data
        """
        ijks = self.inputs_["coordinates"][["i", "j", "k"]].values
        exp_idx = self.inputs_["coordinates"]["id"].values
        if exp_idx is None:
            exp_idx = np.ones(len(ijks))
        exp_idx_uniq, exp_idx = np.unique(exp_idx, return_inverse=True)
        exp_starts = []
        exp_lens = []
        for i_exp, _ in enumerate(exp_idx_uniq):
            # Index peaks by experiment
            curr_exp_idx = exp_idx == i_exp
            exp_starts.append(np.where(curr_exp_idx)[0][0])
            exp_lens.append(curr_exp_idx.sum())
        # convert to array and copy to GPU
        self.d_exp_starts = cupy.asarray(np.array(exp_starts), dtype=cupy.int32)
        self.d_exp_lens = cupy.asarray(np.array(exp_lens), dtype=cupy.int32)
        # set number of experiments
        self.n_exp = len(exp_idx_uniq)
        # set max number of peaks (used to determine number of threads)
        self.max_peaks = max(exp_lens)
        if self.max_peaks > cupy.cuda.runtime.getDeviceProperties(0)['maxThreadsPerBlock']:
            # TODO: break down the foci into multiple blocks if this happens
            raise NotImplementedError(
                "Number of peaks in at least one experiment is higher "
                "than the number of threads per block, which is "
                f"{cupy.cuda.runtime.getDeviceProperties(0)['maxThreadsPerBlock']}. "
                "Please remove these experiments and rerun."
            )

    def _prepare_mask_gpu(self):
        """
        Sets I, J, K coordinates of voxels in mask and a mapping from xyz coordinates
        to voxel index in mask or -1 if voxel is outside mask
        """
        # get ijk of mask voxels
        self.in_mask_voxels = np.array(np.where(self.masker.mask_img.get_fdata()==1)).T
        self.n_voxels_in_mask = self.in_mask_voxels.shape[0]
        self._ALE__n_mask_voxels = self.n_voxels_in_mask # this name is used in nimare
        # create a mapping from xyz coordinates to voxel index in mask
        # or -1 if voxel is outside mask
        # setting the non-mask voxels to -1 serves two purposes:
        # 1. In GPU kernel it is used to identify voxels that are not in mask (and skipped in the loop)
        # 2. In FWE permutation code, in ALE 1D->3D mapping, it points to the last element
        # of the 1D array, which is a np.NaN appended to the end of 1D ALE array on the fly, 
        # and so it makes the background all NaN.
        self.mask_idx_mapping = np.ones(self.masker.mask_img.shape, dtype=int)*-1
        for i in range(self.in_mask_voxels.shape[0]):
            self.mask_idx_mapping[self.in_mask_voxels[i,0], self.in_mask_voxels[i,1], self.in_mask_voxels[i,2]] = i
        self.d_mask_idx_mapping = cupy.asarray(self.mask_idx_mapping, dtype=cupy.int32) # the GPU copy

class DeviceALE(DeviceMixin, ALE):
    def _fit(self, dataset, use_cpu=False):
        """Perform coordinate-based meta-analysis on dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset to analyze.
        use_cpu : :obj:`bool`, optional
            Whether to use CPU instead of GPU. Default is True,
            because the benefit of GPU parallelization is more clear
            in running a large number of ALEs (e.g. in permutations of
            Monte Carlo FWE correction) than in running a single ALE.
        sigma_scale : :obj:`float`, optional
            Scaling factor for kernel sigma. Default is 1.0.
        """
        if use_cpu:
            if self.sigma_scale != 1.0:
                LGR.warn(
                    "sigma_scale is not used when use_cpu is True."
                )
            return super()._fit(dataset)
        # this code is copied from nimare.meta.cbma.base.CBMAEstimator._fit
        # from version 0.2.0 and is modified to run on GPU
        self.dataset = dataset
        self.masker = self.masker or dataset.masker

        if not isinstance(self.masker, NiftiMasker):
            raise ValueError(
                f"A {type(self.masker)} mask has been detected. "
                "Only NiftiMaskers are allowed for this Estimator."
            )

        self.null_distributions_ = {}

        # prepare coordinates data
        self._collect_inputs(self.dataset)
        self._preprocess_input(self.dataset) # will calculate ijks

        # prepare GPU data
        self._set_kernels_gpu()
        self._set_exp_indexing_gpu()
        self._prepare_mask_gpu()

        # allocated memory for MA maps on GPU
        d_ma_values = cupy.zeros((1, self.n_exp, self.n_voxels_in_mask), dtype=self.d_float)
        # copy ijks to GPU
        d_ijks = cupy.asarray(
            self.inputs_["coordinates"][["i", "j", "k"]].values[np.newaxis, :, :], # add a new axis for the single (true) permutation
            dtype=cupy.int32)

        compute_ma_gpu[
            (1, self.n_exp),
            (self.max_peaks,)
        ](
            d_ma_values, d_ijks,
            self.d_kernels, self.d_exp_starts, self.d_exp_lens,
            self.d_mask_idx_mapping,
            self.masker.mask_img.shape[0], self.masker.mask_img.shape[1], self.masker.mask_img.shape[2],
            self.mid, self.mid+1
        )
        # copy MA maps to CPU
        ma_values = d_ma_values.get().squeeze()
        # calculate ALE and copy to CPU
        # reuse the same array in each batch to save memory
        stat_values = \
            (1-cupy.prod((1 - d_ma_values), axis=1)).get().squeeze()
        
        # convert MA maps to sparse format so that it is compatible
        # with the original code
        # following meta.utils.compute_ale_ma
        # TODO: this is a significant bottleneck
        # find a more efficient way to do this
        all_exp = []
        all_coords = []
        all_data = []
        for i_exp in range(self.n_exp):
            nonzero_idx = np.where(ma_values[i_exp] > 0) # which in-mask voxels have non-zero MA in this experiment
            nonzero_ijk = self.in_mask_voxels[nonzero_idx].T # what are their ijk coordinates
            all_exp.append(np.full(nonzero_idx[0].shape[0], i_exp))
            all_coords.append(nonzero_ijk)
            all_data.append(ma_values[i_exp, nonzero_idx])
        exp = np.hstack(all_exp)
        coords = np.vstack((exp.flatten(), np.hstack(all_coords)))
        data = np.hstack(all_data).flatten()
        ma_maps_shape = (self.n_exp,)+self.masker.mask_img.shape # called kernel_shape in nimare
        ma_values = sparse.COO(coords, data, shape=ma_maps_shape)

        # Determine null distributions for summary stat (OF) to p conversion
        self._determine_histogram_bins(ma_values)
        if self.null_method.startswith("approximate"):
            self._compute_null_approximate(ma_values)

        elif self.null_method == "montecarlo":
            raise NotImplementedError("Monte Carlo null method not implemented for GPU.")

        else:
            raise NotImplementedError("Reduced Monte Carlo null method not implemented for GPU.")

        p_values, z_values = self._summarystat_to_p(stat_values, null_method=self.null_method)

        maps = {"stat": stat_values, "p": p_values, "z": z_values}
        description = self._generate_description()
        return maps, {}, description

    def correct_fwe_montecarlo(
        self,
        result,
        voxel_thresh=0.001,
        n_iters=10000,
        n_cores=1,
        batch_size=1,
        vfwe_only=False,
        keep_null_ales=False,
    ):
        # most of this code is copied from nimare.cbma.base.Base.correct_fwe_montecarlo
        # from version 0.2.0 and is modified to run on GPU
        """
        Perform FWE correction using the max-value permutation method on a GPU
        device.

        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Result object from a CBMA meta-analysis.
        voxel_thresh : :obj:`float`, optional
            Cluster-defining p-value threshold. Default is 0.001.
        n_iters : :obj:`int`, optional
            Number of iterations to build the voxel-level, cluster-size, and cluster-mass FWE
            null distributions. Default is 10000.
        n_cores : :obj:`int`, optional
            Number of CPU cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is 1.
            ** There must be one GPU device available to each CPU core. **
        batch_size: :obj:`int`, optional
            Number of permutations to run in each batch on GPU. Default is 1.
            Generally a larger batch size is more efficient, but requires more
            GPU memory. With larger number of experiments memory requirement
            increases and the maximum possible batch size decreases.
            For best performance, start with a higher batch size (e.g. 200)
            and gradually decrease it until OutOfMemoryError is not raised.
        vfwe_only : :obj:`bool`, optional
            If True, only calculate the voxel-level FWE-corrected maps. Voxel-level correction
            can be performed very quickly if the Estimator's ``null_method`` was "montecarlo".
            Default is False.
        keep_null_ales : :obj:`bool`, optional
            Keeps the null ALE maps in the ``null_ales`` attribute for debugging.
            Default is False.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method:

            -   ``logp_desc-size_level-cluster``: Cluster-level FWE-corrected ``-log10(p)`` map
                based on cluster size. This was previously simply called "logp_level-cluster".
                This array is **not** generated if ``vfwe_only`` is ``True``.
            -   ``logp_desc-mass_level-cluster``: Cluster-level FWE-corrected ``-log10(p)`` map
                based on cluster mass. According to :footcite:t:`bullmore1999global` and
                :footcite:t:`zhang2009cluster`, cluster mass-based inference is more powerful than
                cluster size.
                This array is **not** generated if ``vfwe_only`` is ``True``.
            -   ``logp_level-voxel``: Voxel-level FWE-corrected ``-log10(p)`` map.
                Voxel-level correction is generally more conservative than cluster-level
                correction, so it is only recommended for very large meta-analyses
                (i.e., hundreds of studies), per :footcite:t:`eickhoff2016behavior`.
        description_ : :obj:`str`
            A text description of the correction procedure.

        Notes
        -----
        If ``vfwe_only`` is ``False``, this method adds three new keys to the
        ``null_distributions_`` attribute:

            -   ``values_level-voxel_corr-fwe_method-montecarlo``: The maximum summary statistic
                value from each Monte Carlo iteration. An array of shape (n_iters,).
            -   ``values_desc-size_level-cluster_corr-fwe_method-montecarlo``: The maximum cluster
                size from each Monte Carlo iteration. An array of shape (n_iters,).
            -   ``values_desc-mass_level-cluster_corr-fwe_method-montecarlo``: The maximum cluster
                mass from each Monte Carlo iteration. An array of shape (n_iters,).

        Examples
        --------
        >>> meta = dALE()
        >>> result = meta.fit(dset)
        >>> corrector = FWECorrector(method='montecarlo', voxel_thresh=0.01,
                                     n_iters=5, n_cores=1)
        >>> cresult = corrector.transform(result)
        """
        if not hasattr(self, "d_mask_idx_mapping"):
            # in case .fit has been called on CPU
            # this has not been done yet:
            # prepare GPU data
            self._set_kernels_gpu()
            self._set_exp_indexing_gpu()
            self._prepare_mask_gpu()

        stat_values = result.get_map("stat", return_type="array")

        if vfwe_only and (self.null_method == "montecarlo"):
            LGR.info("Using precalculated histogram for voxel-level FWE correction.")

            # Determine p- and z-values from stat values and null distribution.
            p_vfwe_values = nullhist_to_p(
                stat_values,
                self.null_distributions_["histweights_level-voxel_corr-fwe_method-montecarlo"],
                self.null_distributions_["histogram_bins"],
            )

        else:
            if vfwe_only:
                LGR.warn(
                    "In order to run this method with the 'vfwe_only' option, "
                    "the Estimator must use the 'montecarlo' null_method. "
                    "Running permutations from scratch."
                )

            null_xyz = vox2mm(
                np.vstack(np.where(self.masker.mask_img.get_fdata())).T,
                self.masker.mask_img.affine,
            )

            n_cores = _check_ncores(n_cores)

            # Identify summary statistic corresponding to intensity threshold
            ss_thresh = self._p_to_summarystat(voxel_thresh)


            # Generate random coordinates of all permutations and foci
            # in ijk space
            rand_idx = np.random.choice(
                null_xyz.shape[0],
                size=(self.inputs_["coordinates"].shape[0], n_iters),
            )
            rand_xyz = null_xyz[rand_idx, :]
            iter_xyzs = np.split(rand_xyz, rand_xyz.shape[1], axis=1)
            iter_ijks = mm2vox(np.array(iter_xyzs), self.masker.mask_img.affine).squeeze(axis=2)
            

            # Define connectivity matrix for cluster labeling
            d_conn = cupy_ndimage.generate_binary_structure(rank=3, connectivity=1) # used in the loop
            conn = ndimage.generate_binary_structure(rank=3, connectivity=1) # used at the end

            # initialize ALE maps of batch on CPU
            # (only including voxels in mask)
            # this will be overwritten in each batch
            ale_tmp = np.ones((batch_size, self.n_voxels_in_mask), dtype=self.c_float)
            # allocated memory for MA maps of each batch permutations on GPU
            # this will be overwritten in each batch
            d_ma_tmp = cupy.zeros((batch_size, self.n_exp, self.n_voxels_in_mask), dtype=self.d_float)

            if keep_null_ales:
                self.null_distributions_['ale'] = np.ones((n_iters, self.n_voxels_in_mask), dtype=self.c_float)


            fwe_voxel_max = np.zeros(n_iters)
            fwe_cluster_size_max = np.zeros(n_iters)
            fwe_cluster_mass_max = np.zeros(n_iters)

            # Run permutations on GPU
            n_batches = int(np.ceil(n_iters / batch_size))
            for batch_idx in tqdm(range(n_batches)): ## batches are run serially
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, n_iters)
                batch_n_iters = batch_end - batch_start
                d_batch_ijks = cupy.asarray(iter_ijks[batch_start:batch_end], dtype=cupy.int32)
                # initialze ma maps as zeros
                d_ma_tmp[:, :, :] = 0
                compute_ma_gpu[
                    (batch_n_iters, self.n_exp),
                    (self.max_peaks,)
                ](
                    d_ma_tmp, d_batch_ijks,
                    self.d_kernels, self.d_exp_starts, self.d_exp_lens,
                    self.d_mask_idx_mapping,
                    self.masker.mask_img.shape[0], self.masker.mask_img.shape[1], self.masker.mask_img.shape[2],
                    self.mid, self.mid+1
                )
                # calculate ALE of current batch
                ale_tmp[:, :] = \
                        (1-cupy.prod(cupy.subtract(1, d_ma_tmp, out=d_ma_tmp), axis=1)).get()[:batch_n_iters]
                
                if keep_null_ales:
                    self.null_distributions_['ale'][batch_start:batch_end, :] = ale_tmp
                
                # get fwe_voxel_max, fwe_cluster_size_max and fwe_cluster_mass_max
                # for each permutation in batch
                for i_iter in range(batch_n_iters):
                    iter_ale_map = ale_tmp[i_iter, :]
                    # Voxel-level inference
                    fwe_voxel_max[i_iter+batch_start] = np.max(iter_ale_map)

                    if vfwe_only:
                        fwe_cluster_size_max[i_iter+batch_start], fwe_cluster_mass_max[i_iter+batch_start] = None, None
                    else:
                        # Cluster-level inference
                        # convert 1D in-mask ALE array to 3D image data
                        # instead of using the slow self.masker.inverse_transform
                        # using much faster numpy indexing
                        # here we append a NaN at the end of 1D ALE to fill background (index -1) with NaNs
                        iter_ale_map = np.append(iter_ale_map, np.NaN)[self.mask_idx_mapping]
                        # get cluster measures
                        fwe_cluster_size_max[i_iter+batch_start], fwe_cluster_mass_max[i_iter+batch_start] = _calculate_cluster_measures(
                            iter_ale_map, ss_thresh, d_conn, tail="upper"
                        )
            # Free some GPU memory
            del d_batch_ijks, d_ma_tmp
            mempool = cupy.get_default_memory_pool()
            mempool.free_all_blocks()

            ## the rest of the code is unmodified from nimare.meta.cbma.base.Base.correct_fwe_montecarlo

            if not vfwe_only:
                # Cluster-level FWE
                # Extract the summary statistics in voxel-wise (3D) form, threshold, and
                # cluster-label
                thresh_stat_values = self.masker.inverse_transform(stat_values).get_fdata()
                thresh_stat_values[thresh_stat_values <= ss_thresh] = 0
                labeled_matrix, _ = ndimage.label(thresh_stat_values, conn)

                cluster_labels, idx, cluster_sizes = np.unique(
                    labeled_matrix,
                    return_inverse=True,
                    return_counts=True,
                )
                assert cluster_labels[0] == 0

                # Cluster mass-based inference
                cluster_masses = np.zeros(cluster_labels.shape)
                for i_val in cluster_labels:
                    if i_val == 0:
                        cluster_masses[i_val] = 0

                    cluster_mass = np.sum(thresh_stat_values[labeled_matrix == i_val] - ss_thresh)
                    cluster_masses[i_val] = cluster_mass

                p_cmfwe_vals = null_to_p(cluster_masses, fwe_cluster_mass_max, "upper")
                p_cmfwe_map = p_cmfwe_vals[np.reshape(idx, labeled_matrix.shape)]

                p_cmfwe_values = np.squeeze(
                    self.masker.transform(
                        nib.Nifti1Image(p_cmfwe_map, self.masker.mask_img.affine)
                    )
                )
                logp_cmfwe_values = -np.log10(p_cmfwe_values)
                logp_cmfwe_values[np.isinf(logp_cmfwe_values)] = -np.log10(np.finfo(float).eps)
                z_cmfwe_values = p_to_z(p_cmfwe_values, tail="one")

                # Cluster size-based inference
                cluster_sizes[0] = 0  # replace background's "cluster size" with zeros
                p_csfwe_vals = null_to_p(cluster_sizes, fwe_cluster_size_max, "upper")
                p_csfwe_map = p_csfwe_vals[np.reshape(idx, labeled_matrix.shape)]

                p_csfwe_values = np.squeeze(
                    self.masker.transform(
                        nib.Nifti1Image(p_csfwe_map, self.masker.mask_img.affine)
                    )
                )
                logp_csfwe_values = -np.log10(p_csfwe_values)
                logp_csfwe_values[np.isinf(logp_csfwe_values)] = -np.log10(np.finfo(float).eps)
                z_csfwe_values = p_to_z(p_csfwe_values, tail="one")

                self.null_distributions_[
                    "values_desc-size_level-cluster_corr-fwe_method-montecarlo"
                ] = fwe_cluster_size_max
                self.null_distributions_[
                    "values_desc-mass_level-cluster_corr-fwe_method-montecarlo"
                ] = fwe_cluster_mass_max

            # Voxel-level FWE
            LGR.info("Using null distribution for voxel-level FWE correction.")
            p_vfwe_values = null_to_p(stat_values, fwe_voxel_max, tail="upper")
            self.null_distributions_[
                "values_level-voxel_corr-fwe_method-montecarlo"
            ] = fwe_voxel_max

        z_vfwe_values = p_to_z(p_vfwe_values, tail="one")
        logp_vfwe_values = -np.log10(p_vfwe_values)
        logp_vfwe_values[np.isinf(logp_vfwe_values)] = -np.log10(np.finfo(float).eps)

        if vfwe_only:
            # Return unthresholded value images
            maps = {
                "logp_level-voxel": logp_vfwe_values,
                "z_level-voxel": z_vfwe_values,
            }

        else:
            # Return unthresholded value images
            maps = {
                "logp_level-voxel": logp_vfwe_values,
                "z_level-voxel": z_vfwe_values,
                "logp_desc-size_level-cluster": logp_csfwe_values,
                "z_desc-size_level-cluster": z_csfwe_values,
                "logp_desc-mass_level-cluster": logp_cmfwe_values,
                "z_desc-mass_level-cluster": z_cmfwe_values,
            }

        if vfwe_only:
            description = (
                "Family-wise error correction was performed using a voxel-level Monte Carlo "
                "procedure. "
                "In this procedure, null datasets are generated in which dataset coordinates are "
                "substituted with coordinates randomly drawn from the meta-analysis mask, and "
                "the maximum summary statistic is retained. "
                f"This procedure was repeated {n_iters} times to build a null distribution of "
                "summary statistics."
            )
        else:
            description = (
                "Family-wise error rate correction was performed using a Monte Carlo procedure. "
                "In this procedure, null datasets are generated in which dataset coordinates are "
                "substituted with coordinates randomly drawn from the meta-analysis mask, and "
                "maximum values are retained. "
                f"This procedure was repeated {n_iters} times to build null distributions of "
                "summary statistics, cluster sizes, and cluster masses. "
                "Clusters for cluster-level correction were defined using edge-wise connectivity "
                f"and a voxel-level threshold of p < {voxel_thresh} from the uncorrected null "
                "distribution."
            )

        return maps, {}, description
    

class DeviceSCALE(DeviceMixin, SCALE):
    def __init__(self,
                 approach,
                 prob_map=None,
                 xyz=None,
                 sigma_scale=1.0,
                 keep_perm_nulls=False, 
                 use_cpu=False, 
                 batch_size=1,
                 use_mmap=False, 
                 nbits=64,
                 **kwargs):
        """
        Specific (co)activation likelihood estimation on GPU.
        It supports two types of sampling for the null coordinates:
        - 'deterministic' (original approach): the null coordinates 
          are sampled from provided xyz
        - 'probabilistic': the null coordinates are sampled weighted
          by a voxel-wise probability map, e.g. obtained from combining
          kernels of all foci reported in BrainMap.

        Parameters
        ----------
        approach : {'deterministic', 'probabilistic'}
            Approach to use for sampling null coordinates.
        xyz: :obj:`np.ndarray` or None
            xyz coordinates of foci used in the deterministic approach.
            If None, all the voxels within the mask are used. Set to None
            in the probabilistic approach.
        prob_map : :obj:`str`, :class:`~nibabel.nifti1.Nifti1Image`, or None
            Voxel-wise probability map used in the probabilistic approach.
            If None, assumes uniform probability. Set to None in the
            deterministic approach. It is assumed to be a whole-brain
            map, which will then be masked and normalized within the mask.
        sigma_scale : :obj:`float`, optional
            Scaling of the kernel sigma. Default is 1.0.
        keep_perm_nulls : :obj:`bool`, optional
            Whether to keep the null MA maps in the ``null_distributions_`` attribute.
            Default is False.
        use_cpu : :obj:`bool`, optional
        batch_size: :obj:`int`, optional
            Number of permutations to run in each batch on GPU. Default is 1.
            Generally a larger batch size is more efficient, but requires more
            GPU memory. With larger number of experiments memory requirement
            increases and the maximum possible batch size decreases.
            For best performance, start with a higher batch size (e.g. 200)
            and gradually decrease it until OutOfMemoryError is not raised.
        use_mmap: :obj:`bool`, optional
            Whether to use memory-mapped arrays for the null MA maps.
            Default is False. Note that it slows down the permutations
            but can be useful if memory is limited.
        nbits: {32, 64}, optional
            Precision of floating point numbers. Default is 64.
            32-bit precision is faster and uses less memory.
        **kwargs
            Keyword arguments passed to :class:`nimare.meta.cbma.ale.SCALE`,
            including `n_iters` and `n_cores`.
        """
        # TODO: consider creating separate classes
        # for the deterministic and probabilistic approaches
        if xyz is None:
            xyz = np.zeros((1, 3)) # nimare's SCALE expects xyz to be np.ndarray
        super().__init__(xyz, **kwargs)
        self.approach = approach
        if isinstance(prob_map, str):
            prob_map = nib.load(prob_map)
        self.prob_map = prob_map
        self.keep_perm_nulls = keep_perm_nulls
        self.use_cpu = use_cpu
        self.batch_size = batch_size
        self.use_mmap = use_mmap
        self.sigma_scale = sigma_scale
        self.set_dtype(nbits)

    def _prepare_null_sampling(self):
        """Sets the xyz and sampling_prob attributes."""
        if self.approach == 'probabilistic':
            # resample prob_map to mask if needed
            if not np.array_equal(
                self.prob_map.affine, self.masker.mask_img.affine
            ):
                LGR.warn("prob_map and mask do not have the same affine..."
                        "Resampling prob_map to mask affine.")
                self.prob_map = resample_to_img(self.prob_map, self.masker.mask_img, 
                                                interpolation='continuous')
            # mask prob_map
            self.sampling_prob = self.masker.transform(self.prob_map).squeeze()
            # normalize prob_map to a sum probability of 1
            self.sampling_prob /= self.sampling_prob.sum()
            # set xyz to all voxels in mask
            mask_ijks = np.array(np.where(self.masker.mask_img.get_fdata().astype(bool))).T
            self.xyz = vox2mm(mask_ijks, self.masker.mask_img.affine)
        elif self.approach == 'deterministic':
            self.sampling_prob = None
            if self.xyz is None:
                raise ValueError("xyz must be provided in the deterministic approach.")
            
    @use_memmap(LGR, n_files=2)
    def _fit(self, dataset):
        """Perform specific coactivation likelihood estimation meta-analysis on dataset
        on GPU.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset to analyze.
        """
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        self.null_distributions_ = {}

        # prepare coordinates data
        self._collect_inputs(self.dataset)
        self._preprocess_input(self.dataset)

        # prepare GPU data
        self._set_kernels_gpu()
        self._set_exp_indexing_gpu()
        self._prepare_mask_gpu()

        # prepare null sampling
        # (set self.sampling_prob and self.xyz)
        self._prepare_null_sampling()

        # Calculate true MA maps and ALE map
        # allocated memory for MA maps on GPU
        d_ma_values = cupy.zeros((1, self.n_exp, self.n_voxels_in_mask), dtype=self.d_float)
        # copy ijks to GPU
        d_ijks = cupy.asarray(
            self.inputs_["coordinates"][["i", "j", "k"]].values[np.newaxis, :, :], # add a new axis for the single (true) permutation
            dtype=cupy.int32)

        compute_ma_gpu[
            (1, self.n_exp),
            (self.max_peaks,)
        ](
            d_ma_values, d_ijks,
            self.d_kernels, self.d_exp_starts, self.d_exp_lens,
            self.d_mask_idx_mapping,
            self.masker.mask_img.shape[0], self.masker.mask_img.shape[1], self.masker.mask_img.shape[2],
            self.mid, self.mid+1
        )
        # copy MA maps to CPU
        # save ALE and MA as attributes for debugging
        self.ma_values = d_ma_values.get().squeeze()
        # calculate ALE and copy to CPU
        # reuse the same array in each batch to save memory
        self.stat_values = \
            (1-cupy.prod((1 - d_ma_values), axis=1)).get().squeeze()
                
        
        # convert MA maps to sparse format so that it is compatible
        # with the original code
        # following meta.utils.compute_ale_ma
        # TODO: this is a significant bottleneck
        # find a more efficient way to do this
        all_exp = []
        all_coords = []
        all_data = []
        for i_exp in range(self.n_exp):
            nonzero_idx = np.where(self.ma_values[i_exp] > 0) # which in-mask voxels have non-zero MA in this experiment
            nonzero_ijk = self.in_mask_voxels[nonzero_idx].T # what are their ijk coordinates
            all_exp.append(np.full(nonzero_idx[0].shape[0], i_exp))
            all_coords.append(nonzero_ijk)
            all_data.append(self.ma_values[i_exp, nonzero_idx])
        exp = np.hstack(all_exp)
        coords = np.vstack((exp.flatten(), np.hstack(all_coords)))
        data = np.hstack(all_data).flatten()
        ma_maps_shape = (self.n_exp,)+self.masker.mask_img.shape # called kernel_shape in nimare
        ma_values = sparse.COO(coords, data, shape=ma_maps_shape)

        # free GPU memory
        del d_ijks, d_ma_values
        mempool = cupy.get_default_memory_pool()
        mempool.free_all_blocks()

        # Determine bins for null distribution histogram
        max_ma_values = ma_values.max(axis=[1, 2, 3]).todense()

        max_poss_ale = self._compute_summarystat_est(max_ma_values)
        self.null_distributions_["histogram_bins"] = np.round(
            np.arange(0, max_poss_ale + 0.001, 0.0001), 4
        )

        del ma_values

        # Calculate null ALE maps for p-value calculation
        # sample random coordinates from self.xyz and 
        # transform them to ijk space
        rand_idx = np.random.choice(
            self.xyz.shape[0],
            size=(self.inputs_["coordinates"].shape[0], self.n_iters),
            p=self.sampling_prob
        )
        rand_xyz = self.xyz[rand_idx, :]
        iter_xyzs = np.split(rand_xyz, rand_xyz.shape[1], axis=1)
        iter_ijks = mm2vox(np.array(iter_xyzs), self.masker.mask_img.affine).squeeze(axis=2)

        # initialize ALE maps of batch on CPU
        # (only including voxels in mask)
        if self.use_mmap:
            # in memmap file
            perm_scale_values = np.memmap(
                self.memmap_filenames[1],
                dtype=self.c_float,
                mode="w+",
                shape=(self.n_iters, self.n_voxels_in_mask),
            )
        else:
            # in RAM
            perm_scale_values = np.zeros((self.n_iters, self.n_voxels_in_mask), dtype=self.c_float)
        # allocated memory for MA maps of each batch permutations on GPU
        # this will be overwritten in each batch
        d_ma_tmp = cupy.zeros((self.batch_size, self.n_exp, self.n_voxels_in_mask), dtype=self.d_float)

        # Run permutations on GPU
        n_batches = int(np.ceil(self.n_iters / self.batch_size))
        for batch_idx in tqdm(range(n_batches)): ## batches are run serially
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, self.n_iters)
            batch_n_iters = batch_end - batch_start
            d_batch_ijks = cupy.asarray(iter_ijks[batch_start:batch_end], dtype=cupy.int32)
            # initialze ma maps as zeros
            d_ma_tmp[:, :, :] = 0
            compute_ma_gpu[
                        (batch_n_iters, self.n_exp),
                        (self.max_peaks,)
                ](
                    d_ma_tmp, d_batch_ijks,
                    self.d_kernels, self.d_exp_starts, self.d_exp_lens,
                    self.d_mask_idx_mapping,
                    self.masker.mask_img.shape[0], self.masker.mask_img.shape[1], self.masker.mask_img.shape[2],
                    self.mid, self.mid+1
                )
            # calculate ALE 
            # reuse the same array in each batch to save memory
            perm_scale_values[batch_start:batch_end, :] = \
                    (1-cupy.prod(cupy.subtract(1, d_ma_tmp, out=d_ma_tmp), axis=1)).get()[:batch_n_iters]

        # Free some GPU memory
        del d_batch_ijks, d_ma_tmp
        mempool = cupy.get_default_memory_pool()
        mempool.free_all_blocks()


        p_values, z_values = self._scale_to_p(self.stat_values, perm_scale_values)
        if self.keep_perm_nulls:
            self.null_distributions_["perm_scale_values"] = perm_scale_values
        else:
            if isinstance(perm_scale_values, np.memmap):
                LGR.debug(f"Closing memmap at {perm_scale_values.filename}")
                perm_scale_values._mmap.close()
            del perm_scale_values

        logp_values = -np.log10(p_values)
        logp_values[np.isinf(logp_values)] = -np.log10(np.finfo(float).eps)

        # Write out unthresholded value images
        maps = {"stat": self.stat_values, "logp": logp_values, "z": z_values}
        description = self._generate_description()

        return maps, {}, description

    def _scale_to_p(self, stat_values, scale_values, 
                    vox_batch_size=1000, n_elements_per_block=1000,
                    use_cpu=False):
        # added this temporarily to show tqdm for troubleshooting
        """Compute p- and z-values.

        Parameters
        ----------
        stat_values : (V) array
            ALE values.
        scale_values : (I x V) array
            Permutation ALE values.
        vox_batch_size : :obj:`int`, optional
            Number of voxels to process in each batch. Default is 1000.
        n_elements_per_block : :obj:`int`, optional
            Number of elements (observations) processed in each GPU block. 
            Corresponds to the number of threads in each block, and must
            be below the max number of threads per block on the GPU device
            (usually 1024). Default is 1000.

        Returns
        -------
        p_values : (V) array
        z_values : (V) array

        Notes
        -----
        This method also uses the "histogram_bins" element in the null_distributions_ attribute.
        """
        if self.use_cpu:
            # note that the results are not going to be identical
            # (with small number of permutations)
            # because of different treatment of min p-value
            return super()._scale_to_p(stat_values, scale_values)
        n_vox = scale_values.shape[1]
        n_perm = scale_values.shape[0]
        # calculate number of voxel batches and number of GPU blocks
        # needed per permutation
        n_batches = int(np.ceil(n_vox / vox_batch_size)) # batches of voxels
        ## make sure n_elements_per_block is not > n_perm
        n_elements_per_block = min(n_elements_per_block, n_perm)
        n_blocks_per_perm = int(np.ceil(n_perm / n_elements_per_block))
        # set 0s to -1 to skip them in histogram calculations
        # (number of zeros are calculated separately below)
        # I am using -1 instead of NaN because
        # not sure if NaNs can be handled consistently
        # on GPU, i.e., np.NaN float value might be different
        # and I wouldn't trust it to do a value == np.NaN check
        # on the GPU kernel
        scale_values[scale_values==0] = -1.0
        # count number of zeros in the permutations of each voxel
        n_zeros = (scale_values==-1.0).sum(axis=0)
        # define histogram bins
        n_bins = self.null_distributions_["histogram_bins"].shape[0]
        hist_min = np.min(self.null_distributions_["histogram_bins"])
        hist_max = np.max(self.null_distributions_["histogram_bins"])
        # define bin edges: note that although the null value of an example 
        # voxel (first voxel)  is included in the input of this function it 
        # is not needed to recalculate it for every voxel, because when number
        # of bins is fixed and range is provided _get_bin_edges is only
        # going to get the linspace of range and via _get_outer_edge do some
        # checks on the range, and if (a.min(), a.max()) is outside the range 
        # raise an error
        bin_edges = _get_bin_edges(scale_values[:, 0], self.null_distributions_["histogram_bins"], (hist_min, hist_max), None)[0]
        bin_edges = cupy.asarray(bin_edges) # copy to GPU
        # initialize voxel histograms (of each batch) on GPU as zeros
        voxel_hists = cupy.zeros((vox_batch_size, n_bins), dtype=cupy.int32)
        # all_voxel_hists = np.zeros((n_vox, n_bins), dtype=np.int32) # uncomment to keep all hists
        p_values = np.zeros(n_vox)
        # get mempool for GPU memory management
        mempool = cupy.get_default_memory_pool() 
        # keep track of smallest possible p-value across batches
        smallest_value = np.inf
        for i_batch in tqdm(range(n_batches)):
            # reset histograms to all 0s
            voxel_hists.fill(0)
            vox_start = vox_batch_size*i_batch
            vox_end = min(vox_start+vox_batch_size, n_vox)
            curr_batch_size = vox_end - vox_start
            # set the number of zeros in each voxel of this batch
            voxel_hists[:curr_batch_size, 0] = cupy.asarray(n_zeros[vox_start:vox_end], dtype=cupy.int32)
            # copy permutation ALE values of this batch to GPU
            # and reshape it to (n_vox, n_perm)
            voxel_nulls = cupy.asarray(scale_values[:, vox_start:vox_end].T, dtype=self.d_float)
            # call the historgram calculation kernel on GPU
            # with rows (voxels) distributed across grid.x
            # and observations (permutations) distributed across
            # grid.y and block.x
            compute_hist[(curr_batch_size, n_blocks_per_perm,), (n_elements_per_block,)](
                voxel_nulls, 
                bin_edges,
                n_bins, 
                voxel_hists[:, 1:], # first bin is excluded as it includes n of zeros
            )
            # copy histogram to CPU (and execute the kernel)
            voxel_hists_ = voxel_hists.get()
            # calculate p-values, without setting the min
            # p-value to smallest value of null distribution
            # as null distribution and its smallest value is
            # different in each batch of voxels
            p_values[vox_start:vox_end], batch_smallest_value = nullhist_to_p(
                stat_values[vox_start:vox_end],
                voxel_hists_[:curr_batch_size,:].T,
                self.null_distributions_["histogram_bins"],
                is_batch=True
            )
            if batch_smallest_value < smallest_value:
                smallest_value = batch_smallest_value
            # all_voxel_hists[vox_start:vox_end, :] = voxel_hists_[:curr_batch_size, :] # uncomment to keep all hists
            # free GPU memory
            del voxel_nulls
            mempool.free_all_blocks()
        # set the min p-value to smallest value of null distribution
        p_values = np.maximum(p_values, smallest_value)
        # calculate z-values
        z_values = p_to_z(p_values, tail="one")
        return p_values, z_values