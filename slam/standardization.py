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

from copy import deepcopy

from sklearn import preprocessing


def standardize(X, robust=False):
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
    if robust:
        scaler = preprocessing.StandardScaler()
        scaler.scale_ = np.diff(np.nanpercentile(X, (16, 84), axis=0)) / 2.
        scaler.mean_ = np.nanmedian(X, axis=0)
    else:
        scaler = preprocessing.StandardScaler().fit(X)

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
