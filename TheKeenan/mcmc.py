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
stablechain_corrcoef_threshold = 0.1


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
                       return_chain=False, mcmc_run_max_iter=3,
                       prompt=None, *args, **kwargs):
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

    # state of this estimate
    state_inbounds = True
    state_mcc = True
    state_exit = False

    # burn in
    pos0 = [theta0 + np.random.uniform(-1, 1, size=(len(theta0),)) * 0.001
            for _ in range(n_walkers)]
    pos, prob, rstate = sampler.run_mcmc(pos0, n_burnin)

    pos_best = sampler.flatchain[np.argsort(sampler.flatlnprobability)[-1]]

    # run mcmc
    for i_run in range(mcmc_run_max_iter):
        print("i_run: ", i_run)

        sampler.reset()
        pos, prob, rstate = sampler.run_mcmc(pos, n_run)

        print("@Cham: Current pos, ", pos)
        # do check chains
        # 1> check bounds and concentration
        bad_chain_mask, mm, mbest = flatchain_mean_std_check(
            sampler.flatchain, sampler.flatlnprobability,
            n_run, theta_lb, theta_ub)
        print(bad_chain_mask, mm, mbest)

        if np.any(bad_chain_mask):
            # any bad chain, ignore correlation check
            for i_chain, bad_chain_mask_ in enumerate(bad_chain_mask):
                if bad_chain_mask_:
                    # this is a bad chain, change pos
                    pos_new = mbest * (
                        1. + np.random.uniform(-1., 1.,
                                               size=mbest.shape) * 1.e-5)
                    print("@Cham: chain [%s] is reset %s -> %s" %
                          (i_chain, pos[i_chain], pos_new))
                    pos[i_chain] = pos_new
            state_inbounds = False
            continue
        else:
            state_inbounds = True

        # if no chain is reset, then ...
        # 2> max correlation coefficients
        mcc = flatchain_corrcoef_max(sampler.flatchain, n_run)
        print("mcc: ", mcc)
        if mcc >= stablechain_corrcoef_threshold:
            # unstable chain
            state_mcc = False
            continue
        else:
            # stable chain
            state_mcc = True
            state_exit = True
            break

    print(state_inbounds, state_mcc, state_exit)

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


def flatchain_mean_std_check(fchain, fprob, n_step, theta_lb, theta_ub):
    """ calculate correlation coefficients of chains

    Parameters
    ----------
    fchain : ndarray [n_step*n_chain, n_dim]
        MCMC flatchain
    n_step : int
        number of steps of each chain

    Returns
    -------
    coefs : ndarray [n_chain, n_chain, n_dim]
        the corrcoef between each pair of chains

    """
    n_chain = fchain.shape[0] / n_step
    # n_dim = fchain.shape[1]

    # mean & std
    m, s = flatchain_mean_std(fchain, n_step)
    mm = np.median(m, axis=0)  # median of means of chains
    mbest = fchain[np.argsort(fprob)[-1]]  # largest lnprob value

    # assume that only a few chains are bad
    # if theta between [theta_lb, theta_ub]
    # within 3sigma can hit median(mean) of all chains
    # => this is a good chain
    bad_chain_mask = np.zeros((n_chain,), dtype=bool)
    for i_chain, m_ in enumerate(m):
        if np.all(m_ > theta_lb) and np.all(m_ < theta_ub) \
                and np.all(np.abs(m_ - mbest) < s * 3.):
            continue
        else:
            bad_chain_mask[i_chain] = True

    # return bad chain mask
    # suppose mm is a good position
    return bad_chain_mask, mm, mbest


def flatchain_mean_std(fchain, n_step):
    """ calculate correlation coefficients of chains

    Parameters
    ----------
    fchain : ndarray [n_step*n_chain, n_dim]
        MCMC flatchain
    n_step : int
        number of steps of each chain

    Returns
    -------
    coefs : ndarray [n_chain, n_chain, n_dim]
        the corrcoef between each pair of chains

    """
    n_chain = fchain.shape[0] / n_step
    n_dim = fchain.shape[1]

    m = np.zeros((n_chain, n_dim))
    s = np.zeros((n_chain, n_dim))

    for i in range(n_chain):
        ind_i = np.arange(i * n_step, (i + 1) * n_step)
        m[i] = np.mean(fchain[ind_i], axis=0)
        s[i] = np.std(fchain[ind_i], axis=0)

    return m, s


def flatchain_corrcoef(fchain, n_step):
    """ calculate correlation coefficients of chains

    Parameters
    ----------
    fchain : ndarray [n_step*n_chain, n_dim]
        MCMC flatchain
    n_step : int
        number of steps of each chain

    Returns
    -------
    coefs : ndarray [n_chain, n_chain, n_dim]
        the corrcoef between each pair of chains

    """
    n_chain = fchain.shape[0] / n_step
    n_dim = fchain.shape[1]

    coefs = np.zeros((n_chain, n_chain, n_dim))
    for i in range(n_chain):
        ind_i = np.arange(i * n_step, (i + 1) * n_step)
        for j in range(n_chain):
            ind_j = np.arange(j * n_step, (j + 1) * n_step)
            for k in range(n_dim):
                coefs[i, j, k] = \
                    np.corrcoef(fchain[ind_i, k], fchain[ind_j, k])[1, 0]
    return coefs


def flatchain_corrcoef_mean(fchain, n_step):
    """ calculate correlation coefficients of chains

    Parameters
    ----------
    fchain : ndarray [n_step*n_chain, n_dim]
        MCMC flatchain
    n_step : int
        number of steps of each chain

    Returns
    -------
    coefs : ndarray [n_chain, n_chain, n_dim]
        the corrcoef between each pair of chains

    """
    n_chain = fchain.shape[0] / n_step

    coefs = flatchain_corrcoef(fchain, n_step)

    return np.mean(coefs) - 1. / n_chain


def flatchain_corrcoef_max(fchain, n_step):
    """ calculate correlation coefficients of chains

    Parameters
    ----------
    fchain : ndarray [n_step*n_chain, n_dim]
        MCMC flatchain
    n_step : int
        number of steps of each chain

    Returns
    -------
    coefs : ndarray [n_chain, n_chain, n_dim]
        the corrcoef between each pair of chains

    """
    n_chain = fchain.shape[0] / n_step

    coefs = flatchain_corrcoef(fchain, n_step)
    for idim in range(coefs.shape[2]):
        for ichain in range(n_chain):
            coefs[ichain, ichain, idim] = 0.

    return np.max(np.abs(coefs))
