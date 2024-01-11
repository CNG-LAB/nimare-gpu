import numpy as np
from scipy import ndimage
import cupy
from cupyx.scipy import ndimage as cupy_ndimage


def get_ale_kernel(img, sample_size=None, fwhm=None, sigma_scale=1.0):
    # copied from nimare.meta.utils.get_ale_kernel
    # and removed kernel cropping
    """Estimate 3D Gaussian and sigma (in voxels) for ALE kernel given sample size or fwhm.
    
    Parameters
    ----------
    img : :obj:`nibabel.Nifti1Image`
        Image to use as a template for the kernel.
    sample_size : :obj:`int`, optional
        Sample size. If specified, FWHM will be estimated from sample size.
    fwhm : :obj:`float`, optional
        Full-width half-max for Gaussian kernel in mm.
    sigma_scale : :obj:`float`, optional
        Scaling factor for sigma. Default is 1.0.

    Returns
    -------
    sigma_vox : :obj:`float`
        Sigma of Gaussian kernel in voxels.
    kernel : array_like
        3D Gaussian kernel. 
    """
    if sample_size is not None and fwhm is not None:
        raise ValueError('Only one of "sample_size" and "fwhm" may be specified')
    elif sample_size is None and fwhm is None:
        raise ValueError('Either "sample_size" or "fwhm" must be provided')
    elif sample_size is not None:
        uncertain_templates = (
            5.7 / (2.0 * np.sqrt(2.0 / np.pi)) * np.sqrt(8.0 * np.log(2.0))
        )  # pylint: disable=no-member
        # Assuming 11.6 mm ED between matching points
        uncertain_subjects = (11.6 / (2 * np.sqrt(2 / np.pi)) * np.sqrt(8 * np.log(2))) / np.sqrt(
            sample_size
        )  # pylint: disable=no-member
        fwhm = np.sqrt(uncertain_subjects**2 + uncertain_templates**2)

    fwhm_vox = fwhm / np.sqrt(np.prod(img.header.get_zooms()))
    sigma_vox = (
        fwhm_vox * np.sqrt(2.0) / (np.sqrt(2.0 * np.log(2.0)) * 2.0)
    )  # pylint: disable=no-member
    sigma_vox *= sigma_scale

    data = np.zeros((31, 31, 31))
    mid = int(np.floor(data.shape[0] / 2.0))
    data[mid, mid, mid] = 1.0
    kernel = ndimage.gaussian_filter(data, sigma_vox, mode="constant")
    return sigma_vox, kernel

def _calculate_cluster_measures(arr3d, threshold, conn, tail="upper"):
    """Calculate maximum cluster mass and size for an array using GPU.

    This method assesses both positive and negative clusters.

    Parameters
    ----------
    arr3d : :obj:`numpy.ndarray`
        Unthresholded 3D summary-statistic matrix. This matrix will end up changed in place.
    threshold : :obj:`float`
        Uncorrected summary-statistic thresholded for defining clusters.
    conn : :obj:`numpy.ndarray` of shape (3, 3, 3)
        Connectivity matrix for defining clusters.

    Returns
    -------
    max_size, max_mass : :obj:`float`
        Maximum cluster size and mass from the matrix.
    """
    if tail == "upper":
        arr3d[arr3d <= threshold] = 0
    else:
        arr3d[np.abs(arr3d) <= threshold] = 0
    
    arr3d = cupy.asarray(arr3d)

    # labeled_arr3d = np.empty(arr3d.shape, int)
    labeled_arr3d, _ = cupy_ndimage.label(arr3d>0, conn)

    if tail == "two":
        # Label positive and negative clusters separately
        n_positive_clusters = cupy.max(labeled_arr3d)
        temp_labeled_arr3d, _ = cupy_ndimage.label(arr3d < 0, conn)
        temp_labeled_arr3d[temp_labeled_arr3d > 0] += n_positive_clusters
        labeled_arr3d = labeled_arr3d + temp_labeled_arr3d
        del temp_labeled_arr3d

    clust_sizes = cupy.bincount(labeled_arr3d.flatten())
    clust_vals = cupy.arange(0, clust_sizes.shape[0])

    # Cluster mass-based inference
    max_mass = 0
    for unique_val in clust_vals[1:]:
        ss_vals = cupy.abs(arr3d[labeled_arr3d == unique_val]) - threshold
        max_mass = cupy.maximum(max_mass, cupy.sum(ss_vals))

    # Cluster size-based inference
    clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
    if clust_sizes.size:
        max_size = cupy.max(clust_sizes)
    else:
        max_size = 0

    del arr3d, labeled_arr3d, clust_sizes, clust_vals

    return max_size, max_mass