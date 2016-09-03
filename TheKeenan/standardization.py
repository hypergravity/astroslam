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

from sklearn import preprocessing


def standardize(X):
    """ Standardize X

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
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return scaler, X_scaled


if __name__ == '__main__':
    import numpy as np

    x = np.random.randn(10, 20)
    s, xs = standardize(x)
    print (x, xs)
    print (s, s.mean_, s.scale_)
