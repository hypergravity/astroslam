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
        weight = np.ones_like(X, bool)

    if robust:
        scaler = preprocessing.StandardScaler()
        X_ = np.where(weight > 0, X, np.nan)
        scaler.scale_ = (np.diff(np.nanpercentile(X_, (16, 84), axis=0),
                                 axis=0) / 2.).flatten()
        scaler.scale_ = np.where(scaler.scale_ <= 0, 1., scaler.scale_)
        scaler.mean_ = np.nanmedian(X_, axis=0)
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
