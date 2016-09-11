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
- Implement bayesian estimation of stars

"""

import numpy as np
from emcee import EnsembleSampler

from .predict import predict_spectrum

eps = 1e-10  # Once flux_ivar < eps, these pixels are ignored


def lnlike_gaussian(theta, svrs, flux_obs, flux_ivar, mask):
    """ Gaussian likelihood function

    Parameters
    ----------
    theta : ndarray
        the stellar labels
    flux_obs : ndarray
        the observed stellar spectrum
    flux_ivar : ndarray
        the inverse variance of observed spectrum
    scaler :
        label scaler, i.e., tr_labels_scaler
    svrs : list
        a list of svm.SVR() objects

    Returns
    -------
    Gaussian likelihood function

    NOTE
    ----
    since flux_ivar appears in denominator, 0 values are ignored

    """
    # determine good pixels
    # ind_good = flux_ivar > eps
    # flux_ivar[~ind_good] = eps

    # preprocessing is already done

    # predict spectrum
    flux_pred = predict_spectrum(svrs, theta, mask=mask)

    # Gaussian likelihood
    return - 0.5 * np.nansum((flux_obs - flux_pred) ** 2. * flux_ivar +
                             np.log(2. * np.pi / flux_ivar))


def lnprior_uniform(theta, theta_lb, theta_ub):
    """ loose uniform prior for theta

    Parameters
    ----------
    theta : ndarray
        the stellar labels

    """
    theta = np.array(theta)

    # if np.all(-np.inf < theta) and np.all(theta < np.inf):
    if np.all(theta_lb < theta) and np.all(theta < theta_ub):
        # reasonable theta
        return 0.
    # unreasonable theta
    return -np.inf


def lnprob(theta, svrs, flux_obs, flux_ivar, mask, theta_lb, theta_ub):
    """ posterior probability function

    Parameters
    ----------
    theta : ndarray
        the stellar labels
    scaler :
        label scaler, i.e., tr_labels_scaler
    svrs : list
        a list of svm.SVR() objects
    flux_obs : ndarray
        the observed stellar spectrum
    flux_ivar : ndarray
        the inverse variance of observed spectrum

    Returns
    -------
    Gaussian likelihood function

    NOTE
    ----
    since flux_ivar appears in denominator, 0 values are ignored

    """
    # calculate prior
    lp = lnprior_uniform(theta, theta_lb, theta_ub)

    if not np.isfinite(lp):
        # if prior is unreasonable (-inf), avoiding lnlike computing
        return -np.inf

    # if prior is reasonable
    lp += lnlike_gaussian(theta, svrs, flux_obs, flux_ivar, mask)
    # print("theta: ", theta, "lp: ", lp)
    return lp


def predict_label_mcmc(theta0, svrs, flux_obs, flux_ivar, mask,
                       theta_lb=None, theta_ub=None,
                       n_walkers=10, n_burnin=200, n_run=500, threads=1,
                       return_chain=False,
                       *args, **kwargs):
    n_dim = len(theta0)

    if theta_lb is None:
        theta_lb = np.ones_like(theta0) * -10.
    if theta_ub is None:
        theta_ub = np.ones_like(theta0) * 10.

    # instantiate
    sampler = EnsembleSampler(n_walkers, n_dim, lnprob,
                              args=(svrs, flux_obs, flux_ivar, mask,
                                    theta_lb, theta_ub),
                              threads=threads)  # **kwargs?

    # burn in
    pos0 = [theta0 + np.random.randn(len(theta0)) * 0.001 for _ in
            range(n_walkers)]
    # print(pos0)
    pos, prob, state = sampler.run_mcmc(pos0, n_burnin)

    # run
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, n_run)

    # estimate percentiles
    theta_est_mcmc = np.percentile(sampler.flatchain, [15., 50., 85.], axis=0)

    # format of theta_est_mcmc:
    # array([theta_p15,
    #        theta_p50,
    #        theta_p85])
    # e.g.:
    # array([[ 3.21908185,  5.66655696,  8.99618546],
    #        [ 3.22411158,  5.68827311,  9.08791289],
    #        [ 3.22909087,  5.71157073,  9.17812294]])

    if not return_chain:
        return theta_est_mcmc
    else:
        return theta_est_mcmc, sampler.flatchain
