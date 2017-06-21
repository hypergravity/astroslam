# -*- coding: utf-8 -*-
"""

Author
------
Bo Zhang

Email
-----
bozhang@nao.cas.cn

Created on
----------
- Sun Jun 11 12:00:00 2017

Modifications
-------------
- Sun Jun 11 12:00:00 2017

Aims
----
- utils for processing spectra

"""

import numpy as np


def convolve_mask(mask, kernel_size_coef=.25, kernel_size_limit=(2, 100),
                  sink_region=(200, .5)):
    """
    
    Parameters
    ----------
    mask: array like
        initial mask. True for good pixels, False for bad ones.
    kernel_size_coef:
        the kernel_size/bad_chunk_length coefficient
    kernel_size_limit: tuple
        (lower limit, upper limit)
    sink_region: tuple
        (width, threshold fraction of good pixels in this region)
        if None, pass
        
    Returns
    -------
    convolved mask

    """
    # mask: True for good, False for bad
    # 1. kernel length: at least 2 pixel at most 100 pixels
    mask0 = np.array(mask, bool)
    mask1 = np.array(np.hstack((True, mask0, True)), int)
    mask2 = np.copy(mask0)

    mask1_diff = np.diff(mask1)
    bad_chunks = np.vstack(
        (np.where(mask1_diff < 0)[0], np.where(mask1_diff > 0)[0])).T

    bad_chunks_len = np.round(np.diff(bad_chunks, axis=1) * kernel_size_coef)
    bad_chunks_len = np.where(bad_chunks_len < kernel_size_limit[0],
                              kernel_size_limit[0], bad_chunks_len)
    bad_chunks_len = np.where(bad_chunks_len > kernel_size_limit[1],
                              kernel_size_limit[1], bad_chunks_len)

    bad_chunks_convolved = np.array(
        bad_chunks_len.reshape(-1, 1) * np.array([-1, 1]) + bad_chunks, int)
    bad_chunks_convolved = np.where(
        bad_chunks_convolved < 0, 0, bad_chunks_convolved)
    bad_chunks_convolved = np.where(
        bad_chunks_convolved >= len(mask0), len(mask0), bad_chunks_convolved)

    for i_chunk in range(bad_chunks_convolved.shape[0]):
        mask2[bad_chunks_convolved[i_chunk, 0]:bad_chunks_convolved[
            i_chunk, 1]] = False

    # 2. sink_region: second round mask convolution
    if sink_region is not None:
        ind_min = 0
        ind_max = len(mask0)

        good_frac = np.zeros_like(mask0, float)
        for i in range(ind_max):
            this_start = np.max((ind_min, i - sink_region[0]))
            this_stop = np.min((ind_max, i + sink_region[0]))
            if (this_stop - this_start) < (2 * sink_region[0]):
                if this_start == ind_min:
                    this_stop = this_start + sink_region[0]
                else:
                    this_start = this_stop - sink_region[0]
            good_frac[i] = np.sum(mask0[this_start:this_stop]) / \
                           (this_stop - this_start)
        mask2 = np.where(good_frac < sink_region[1], False, mask2)

    return mask2
