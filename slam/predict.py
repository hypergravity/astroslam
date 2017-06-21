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
from joblib import Parallel, delayed

# from .mcmc import lnlike_gaussian
from .costfunction import chi2_simple_1d
from scipy.optimize import leastsq


def predict_pixel(svr, X_, mask=True):
    """ predict single pixels for a given wavelength

    Parameters
    ----------
    svr: sklearn.svm.SVR instance
        the pixel SVR to diagnostic
    X_: ndarray ( :, ndim )
        test_labels that will be evaluated
    mask: bool
        if True, evaluate
        if False, pass

    Returns
    -------
    y: ndarray
        predicted flux

    """
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
    svrs : list
        a list of svr objects
    X_ : ndarray
        the labels of predicted spectra
    mask : None | bool array
        predict the pixels where mask==True
    scaler : scaler object
        if not None, scale X_ before predictions using this scaler

    Returns
    -------
    ys : ndarray
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
                   flux_scaler=None, ivar_scaler=None, labels_scaler=None,
                   **kwargs):
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

    # scale test_ivar if necessary
    if ivar_scaler is not None:
        test_ivar = ivar_scaler.transform(test_ivar.reshape(1, -1)).flatten()

    # print ("Xshape in predict_labels: ", X0.shape)
    # print costfun_for_label(X0, svrs, test_flux, test_ivar, mask)
    X_pred, ier = leastsq(costfun_for_label, X0,
                          args=(svrs, test_flux, test_ivar, mask), **kwargs)
    # do minimization using Nelder-Mead method [tol=1.e-8 set by user!]
    # X_pred = minimize(costfun_for_label, X0,
    #                   args=(svrs, test_flux, test_ivar, mask),
    #                   method='Nelder-Mead', **kwargs)
    # nll = lambda *args: -lnlike_gaussian(*args)
    # X_pred = minimize(nll, X0,
    #                   args=(svrs, test_flux, test_ivar, mask),
    #                   method='Nelder-Mead', **kwargs)
    print('@Cham: X_init=', X0, 'X_final=', X_pred, 'ier', ier)
    # , 'nit=', X_pred['nit']

    # scale X_pred back if necessary
    if labels_scaler is not None:
        X_pred = labels_scaler.inverse_transform(
            X_pred.reshape(1, -1)).flatten()
    else:
        X_pred = X_pred.flatten()

    return X_pred


def costfun_for_label(X_, svrs, test_flux, test_ivar, mask):
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
    # print ("X_ in costfun_for_label: ", X_)
    X_.reshape(1, -1)
    # default is to use all pixels [True->will be used, False->deprecated]
    if mask is None:
        mask = np.ones((len(test_flux),), dtype=np.bool)

    # default ivar is all 1
    if test_ivar is None:
        test_ivar = np.ones_like(test_flux)
    else:
        test_ivar[test_ivar < 0] = 0.
        # kick more pixels using 0.01 ivar --> NON-PHYSICAL
        # mask = np.logical_and(mask, test_ivar > 0.01 * np.median(test_ivar))

    # do prediction
    pred_flux = predict_spectrum(svrs, X_, mask).astype(np.float)
    # the pred_flux contains nan for mask=False pixels

    # print ("test_flux", test_flux, test_flux.shape)
    # print ("pred_flux", pred_flux, pred_flux.shape)
    # print ("test_ivar", test_ivar, test_ivar.shape)

    # calculate chi2
    # return chi2_simple_1d(test_flux, pred_flux, ivar=test_ivar)
    res = (test_flux.flatten()-pred_flux.flatten())*test_ivar.flatten()
    res[np.isnan(res)] = 0.
    return res


def predict_labels_chi2(tplt_flux, tplt_ivar, tplt_labels, test_flux, test_ivar,
                        n_jobs=1, verbose=False):
    """ a quick search for initial values of test_labels for test_flux

    NOTE
    ----
    this is a nested function

    """

    assert tplt_flux.ndim == 2 and tplt_labels.ndim == 2

    if test_flux.ndim == 1:
        # only one test_flux
        # n_test = 1
        assert tplt_flux.shape[1] == test_flux.shape[0]

        i_min = np.argsort(
            np.nanmean((tplt_flux - test_flux) ** 2. * test_ivar * np.where(
                tplt_ivar > 0, 1., np.nan), axis=1)
        ).flatten()[0]

        return tplt_labels[i_min, :]

    else:
        n_test = test_flux.shape[0]
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(predict_labels_chi2)(
                tplt_flux, tplt_ivar, tplt_labels, test_flux[i, :], test_ivar[i, :])
            for i in range(n_test)
        )

        return np.array(results)


def predict_pixel_for_diagnostic(svr,
                                 test_labels,
                                 labels_scaler=None,
                                 flux_mean_=0.,
                                 flux_scale_=1.):
    """

    Parameters
    ----------
    svrs: list of sklearn.svm.SVR instance
        the pixel SVR to diagnostic
    test_labels: ndarray ( :, ndim )
        test_labels that will be evaluated
    labels_scaler: sklearn.preprocessing.StandardScaler
        the scaler for labels
    flux_scaler: sklearn.preprocessing.StandardScaler
        the scaler for flux

    Returns
    -------
    test_flux

    """
    # transform test labels
    if labels_scaler is not None:
        test_labels = labels_scaler.transform(test_labels)

    # predict pixels
    test_flux = predict_pixel(svr, test_labels, mask=True)[:, None]

    # inverse transform predicted flux
    if flux_mean_ is not None and flux_scale_ is not None:
        test_flux = test_flux * flux_scale_ + flux_mean_

    return test_flux
