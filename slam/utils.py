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
import sys
from collections import OrderedDict, Set, Mapping, deque
from numbers import Number

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


def uniform(tr_labels, bins, n_pick=3, ignore_out=False, digits=8):
    """ make a uniform sample --> index stored in Slam.uniform_good

    Parameters
    ----------
    bins: list of arrays
        bins in each dim
    n_pick: int
        how many to pick in each bin
    ignore_out: bool
        if True, kick stars out of bins
        if False, raise error if there is any star out of bins 
    digits: int
        digits to form string
        
    Examples
    --------
    >>> uniform(data, [np.arange(3000, 6000, 100), np.arange(-1, 5, .2),
    >>>     np.arange(-5, 1, .1)], n_pick=1, ignore_out=False)

    Returns
    -------
    index of selected sub sample

    """

    n_obs, n_dim = tr_labels.shape
    try:
        assert len(bins) == n_dim
    except AssertionError:
        print("@utils.uniform: ", len(bins), n_dim, "don't match")

    # initiate arrays
    uniform_good = np.ones((n_obs,), bool)
    uniform_ind = np.ones_like(tr_labels, int) * np.nan

    # make IDs for bins
    for i_dim in range(n_dim):
        this_bins = bins[i_dim]
        for i_bin in range(len(this_bins) - 1):
            ind = np.logical_and(
                tr_labels[:, i_dim] > this_bins[i_bin],
                tr_labels[:, i_dim] < this_bins[i_bin + 1])
            uniform_ind[ind, i_dim] = i_bin

    # check bins covering all stars
    ind_not_in_bins = np.any(
        np.logical_not(np.isfinite(uniform_ind)), axis=1)
    if np.sum(ind_not_in_bins) > 0:
        if ignore_out:
            print("@utils.uniform: These stars are out of bins and ignored")
            print("i = ", np.where(ind_not_in_bins)[0])
            uniform_good &= np.logical_not(ind_not_in_bins)
        else:
            raise (ValueError(
                "@utils.uniform: bins not wide enough to cover all stars"))

    # make ID string for bins
    fmt = "{{:0{}.0f}}".format(digits)
    uniform_str = []
    for i_obs in range(n_obs):
        str_ = ""
        for i_dim in range(n_dim):
            str_ += fmt.format(uniform_ind[i_obs, i_dim])
        uniform_str.append(str_)
    uniform_str = np.array(uniform_str)

    # unique IDs
    u_str, u_inverse, u_counts = np.unique(
        uniform_str, return_inverse=True, return_counts=True)

    # pick stars from these bins
    ind_bin_need_to_pick = np.where(u_counts > n_pick)[0]
    for _ in ind_bin_need_to_pick:
        ind_in_this_bin = np.where(u_inverse == _)[0]
        np.random.shuffle(ind_in_this_bin)
        uniform_good[ind_in_this_bin[n_pick:]] = False

    print("@utils.uniform: [{}/{}] stars chosen to make a uniform sample!"
          "".format(np.sum(uniform_good), n_obs))

    return dict(uniform_picked=uniform_good,
                uniform_unpicked=np.logical_not(uniform_good),
                uniform_ind=uniform_ind,
                uniform_str=uniform_str,
                uniform_bins=bins,
                n_pick=n_pick,
                digits=digits,
                ignore_out=ignore_out)

unit_scale_dict = dict(
    b=1.,
    kb=1024**-1,
    mb=1024**-2,
    gb=1024**-3,
)


# deprecated
def sizeof(obj, unit='mb', verbose=False, key_removed=None):

    # get scale for unit
    try:
        scale = unit_scale_dict[unit]
    except KeyError:
        print("@sizeof: unit should be in ", unit_scale_dict.keys())

    # get size
    v_dict = OrderedDict()
    for _ in dir(obj):
        v_dict[_] = getsize(obj.__getattribute__(_))

    v_dict['_total'] = np.sum([v_dict[_] for _ in v_dict.keys()])
    for k in v_dict.keys():
        v_dict[k] = np.int(scale * v_dict[k])

    v_dict['_unit'] = unit

    if verbose:
        print(v_dict)

    return v_dict


# ########################################################################### #
# adapted from https://stackoverflow.com/questions/449560/how-do-i-determine-th
# e-size-of-an-object-in-python
# ########################################################################### #

# try: # Python 2
#     zero_depth_bases = (basestring, Number, xrange, bytearray)
#     iteritems = 'iteritems'
# except NameError: # Python 3
zero_depth_bases = (str, bytes, Number, range, bytearray)
iteritems = 'items'


def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    def inner(obj, _seen_ids = set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)

