import numpy as np
from scipy import ndimage


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