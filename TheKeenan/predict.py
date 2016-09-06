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
- Sun Sep 04 16:00:00 2016

Modifications
-------------
- Sun Sep 04 16:00:00 2016

Aims
----
- utils for training SVRs

"""

import numpy as np
from scipy.optimize import minimize

from costfunction import chi2_simple_1d


def predict_pixel(svr, X_, mask=True):
    """ predict single pixels for a given wavelength """
    assert X_.ndim == 2

    # print('mask: ', mask)
    if mask:
        y = svr.predict(X_)  # predicted y is a flatten array
    else:
        y = np.nan

    return y


def predict_spectrum(svrs, X_, mask=None, scaler=None):
    """ predict a single spectrum given a list of svrs & mask

    Parameters
    ----------
    svrs: list
        a list of svr objects
    mask: None | bool array
        predict the pixels where mask==True
    scaler: scaler object
        if not None, scale X_ before predictions using this scaler

    Returns
    -------
    ys: ndarray
        predicted spectra

    """
    if X_.ndim == 1:
        X_ = X_.reshape(1, -1)

    # scale X_ if necessary
    if scaler is not None:
        X_ = scaler.transform(X_)

    # default is to use all pixels
    if mask is None:
        mask = np.ones((len(svrs),), dtype=np.bool)

    # make predictions
    # print('number of true mask: ', np.sum(mask))
    # print('mask len: ', mask.shape)
    ys = [predict_pixel(svr, X_, mask_) for svr, mask_ in zip(svrs, mask)]
    ys = np.array(ys).T

    return ys


def predict_labels(X0, svrs, test_flux, test_ivar=None, mask=None,
                   flux_scaler=None, labels_scaler=None, **kwargs):
    """ predict scaled labels for test_flux

    Parameters
    ----------
    X0 : ndarray (n_test, n_dim)
        initial guess
    svrs: list
        a list of svr objects
    test_flux: ndarray
        test flux
    test_ivar: ndarray
        test ivar
    mask: None | bool array
        predict the pixels where mask==True
    flux_scaler: scaler object
        if not None, scale test_flux before predictions
    labels_scaler: scaler object
        if not None, scale predicted labels back to normal scale

    Returns
    -------
    X_pred: ndarray
        predicted lables (scaled)

    """
    # assert X0 is 2D array
    assert X0.ndim == 2

    # scale test_flux if necessary
    if flux_scaler is not None:
        test_flux = flux_scaler.transform(test_flux.reshape(1, -1)).flatten()

    # do minimization using Nelder-Mead method [tol=1.e-8 set by user!]
    X_pred = minimize(costfun_from_label, X0,
                      args=(svrs, test_flux, test_ivar, mask),
                      method='Nelder-Mead', **kwargs)

    # scale X_pred back if necessary
    if labels_scaler is not None:
        X_pred = labels_scaler.inverse_transform(
            X_pred['x'].reshape(1, -1)).flatten()

    return X_pred


def costfun_from_label(X_, svrs, test_flux, test_ivar, mask):
    """ calculate (ivar weighted) chi2 for a single spectrum

    Parameters
    ----------
    svrs: list
        a list of svr objects
    test_flux: ndarray (n_pix, )
        test flux
    test_ivar: ndarray (n_pix, )
        test ivar
    mask: None | bool array
        predict the pixels where mask==True

    """
    # default is to use all pixels
    if mask is None:
        mask = np.ones((len(test_flux),), dtype=np.bool)

    # default ivar is all 1
    if test_ivar is None:
        test_ivar = np.ones_like(test_flux)
    else:
        test_ivar[test_ivar < 0] = 0.
        # kick more pixels using 0.01 ivar
        mask = np.logical_and(mask, test_ivar > 0.01 * np.median(test_ivar))

    # do prediction
    pred_flux = predict_spectrum(svrs, X_, mask)
    # the pred_flux contains nan for mask=False pixels

    # calculate chi2
    return chi2_simple_1d(test_flux, pred_flux, ivar=test_ivar)
