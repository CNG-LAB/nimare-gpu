import numpy as np
from nimare import utils

def nullhist_to_p(test_values, histogram_weights, histogram_bins, is_batch=False):
    """Return one-sided p-value for test value against null histogram.

    .. versionadded:: 0.0.4

    Parameters
    ----------
    test_values : float or 1D array_like
        Values for which to determine p-value. Can be a single value or a one-dimensional array.
        If a one-dimensional array, it should have the same length as the histogram_weights' last
        dimension.
    histogram_weights : (B [x V]) array
        Histogram weights representing the null distribution against which test_value is compared.
        These should be raw weights or counts, not a cumulatively-summed null distribution.
    histogram_bins : (B) array
        Histogram bin centers. Note that this differs from numpy.histogram's behavior, which uses
        bin *edges*. Histogram bins created with numpy will need to be adjusted accordingly.
    is_batch: bool
        If is part of a batch, skip setting the smallest p-value to the smallest non-zero value
        but return it so that the smallest is set across all batches. Default: False

    Returns
    -------
    p_value : :obj:`float`
        P-value associated with the test value when compared against the null distribution.
        P-values reflect the probability of a test value at or above the observed value if the
        test value was drawn from the null distribution.
        This is a one-sided p-value.
    smallest_value : :obj:`float`
        Smallest non-zero p-value in the null distribution in current batch. 
        Only returned if is_batch is True.
    """
    test_values = np.asarray(test_values)
    return_value = False
    if test_values.ndim == 0:
        return_value = True
        test_values = np.atleast_1d(test_values)
    assert test_values.ndim == 1
    assert histogram_bins.ndim == 1
    assert histogram_weights.shape[0] == histogram_bins.shape[0]
    assert histogram_weights.ndim in (1, 2)
    if histogram_weights.ndim == 2:
        assert histogram_weights.shape[1] == test_values.shape[0]
        voxelwise_null = True
    else:
        histogram_weights = histogram_weights[:, None]
        voxelwise_null = False

    n_bins = len(histogram_bins)
    inv_step = 1 / (histogram_bins[1] - histogram_bins[0])  # assume equal spacing

    # Convert histograms to null distributions
    # The value in each bin represents the probability of finding a test value
    # (stored in histogram_bins) of that value or lower.
    null_distribution = histogram_weights / np.sum(histogram_weights, axis=0)
    null_distribution = np.cumsum(null_distribution[::-1, :], axis=0)[::-1, :]
    null_distribution /= np.max(null_distribution, axis=0)
    null_distribution = np.squeeze(null_distribution)

    smallest_value = np.min(null_distribution[null_distribution != 0])

    p_values = np.ones(test_values.shape)
    idx = np.where(test_values > 0)[0]
    value_bins = utils._round2(test_values[idx] * inv_step)
    value_bins[value_bins >= n_bins] = n_bins - 1  # limit to within null distribution

    # Get p-values by getting the value_bins-th value in null_distribution
    if voxelwise_null:
        # Pair each test value with its associated null distribution
        for i_voxel, voxel_idx in enumerate(idx):
            p_values[voxel_idx] = null_distribution[value_bins[i_voxel], voxel_idx]
    else:
        p_values[idx] = null_distribution[value_bins]

    # ensure p_value in the following range:
    # smallest_value <= p_value <= 1.0
    p_values = np.minimum(p_values, 1.0)
    if not is_batch:
        p_values = np.maximum(smallest_value, p_values)
    if return_value:
        p_values = p_values[0]
    if is_batch:
        return p_values, smallest_value
    else:
        return p_values