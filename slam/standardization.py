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
- Fri Sep 02 13:00:00 2016

Modifications
-------------
- Fri Sep 02 13:00:00 2016

Aims
----
- standardization

"""

import numpy as np
from copy import deepcopy

from sklearn import preprocessing


def standardize(X, weight=None, robust=False):
    """ Standardize X (flux / labels)

    Parameters
    ----------
    X: ndarray
        data array

    Returns
    -------
    scaler: sklearn.StandardScaler
        scaler

    X_scaled: ndarray
        scaled X

    """
    if weight is None:
        weight = np.ones_like(X, int)
    weight = np.logical_and(weight > 0, np.isfinite(X))

    ind_good = weight > 0
    n_good = np.sum(ind_good, axis=0)

    scaler = preprocessing.StandardScaler()
    n_col = X.shape[1]

    scaler.scale_ = np.ones((n_col,), float)
    scaler.mean_ = np.zeros((n_col,), float)

    if robust:
        # estimate using percentiles
        for i_col in range(n_col):
            if n_good[i_col] > 0:
                # at least 1 good pixels
                scaler.mean_[i_col] = (np.nanpercentile(
                    X[ind_good[:, i_col], i_col], 84) + np.nanpercentile(
                    X[ind_good[:, i_col], i_col], 16)) / 2.
                scaler.scale_[i_col] = (np.nanpercentile(
                    X[ind_good[:, i_col], i_col], 84) - np.nanpercentile(
                    X[ind_good[:, i_col], i_col], 16)) / 2.

    else:
        # estimate using mean and std
        for i_col in range(n_col):
            if n_good[i_col] > 0:
                # at least 1 good pixels
                scaler.scale_[i_col] = np.std(X[ind_good[:, i_col], i_col])
                scaler.mean_[i_col] = np.mean(X[ind_good[:, i_col], i_col])

    scaler.scale_ = np.where(scaler.scale_ < 1e-300, 1., scaler.scale_)
    scaler.robust = robust
    X_scaled = scaler.transform(X)
    return scaler, X_scaled


def standardize_ivar(ivar, flux_scaler):
    """ ivar_scaler is copied from flux_scaler, but mean_ is set to be 0
    """
    # copy flux_scaler & generate ivar_scaler
    ivar_scaler = deepcopy(flux_scaler)
    ivar_scaler.mean_ *= 0
    ivar_scaler.scale_ **= -2.  # this is extremely important!
    # transform ivar data
    ivar_scaled = ivar_scaler.transform(ivar)
    return ivar_scaler, ivar_scaled


if __name__ == '__main__':
    import numpy as np

    x = np.random.randn(10, 20)
    s, xs = standardize(x)
    print (x, xs)
    print (s, s.mean_, s.scale_)
