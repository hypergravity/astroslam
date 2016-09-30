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
- Sat Sep 03 12:00:00 2016

Modifications
-------------
- Sat Sep 03 12:00:00 2016

Aims
----
- define cost functions
 - temporally only chi2 is implemented

"""

from __future__ import division

import numpy as np
from scipy.stats import chisquare


__all__ = ['chi2_simple_1d']


def chi2_simple_1d(spec_obs, spec_pred, ivar=None):
    """ Calculate ivar-weighted chi-square for two spectra

    Parameters
    ----------
    spec_obs: array-like
        observed flux

    spec_pred: array-like
        predicted flux

    ivar: array-like or None
        inverse variance / weight of pixels
        if None, assume pixels are even-weighted

    """
    if ivar is None:
        # ivar not specified
        chi2_value = np.nansum((np.array(spec_obs).flatten() - np.array(
            spec_pred).flatten()) ** 2.)
    else:
        # ivar specified
        chi2_value = np.nansum((np.array(spec_obs).flatten() - np.array(
            spec_pred).flatten()) ** 2. * np.array(ivar).flatten())
    # print('chi2: ', chi2_value)
    return chi2_value


def chi2(a, b):
    c = chisquare([1, 2, 3], [1, 1, 1])
    return c.statistic
