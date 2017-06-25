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


__all__ = ['lnlike_gaussian', 'lnprior_uniform', 'lnprob',
           'predict_label_mcmc', 'predict_spectrum',
           'theta_between', 'check_chains', 'sampler_mcc']

eps = 1e-10  # Once flux_ivar < eps, these pixels are ignored
stablechain_corrcoef_threshold = 0.4


def lnlike_gaussian(theta, svrs, flux_obs, flux_ivar, mask):
    """ Gaussian likelihood function

    Parameters
    ----------
    theta: ndarray
        the stellar labels
    svrs: list
        a list of sklearn.svm.SVR objects
    flux_obs: ndarray
        the observed stellar spectrum
    flux_ivar: ndarray
        the inverse variance of observed spectrum
    mask:
        label scaler, i.e., tr_labels_scaler


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
    theta: ndarray
        the stellar labels
    theta_lb: 2-element ndarray
        the lower bound of theta
    theta_ub: 2-element ndarray
        the upper bound of theta

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
    svrs: list
        a list of sklearn.svm.SVR objects
    flux_obs : ndarray
        the observed stellar spectrum
    flux_ivar : ndarray
        the inverse variance of observed spectrum
    mask: bool array



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
                       return_chain=False, mcmc_run_max_iter=5, mcc=0.4,
                       prompt=None, **kwargs):
    """ predict labels using emcee MCMC """
    # theta length
    n_dim = len(theta0)

    # default theta lower/upper bounds
    if theta_lb is None:
        theta_lb = np.ones_like(theta0) * -10.
    if theta_ub is None:
        theta_ub = np.ones_like(theta0) * 10.

    # instantiate EnsambleSampler
    sampler = EnsembleSampler(n_walkers, n_dim, lnprob,
                              args=(svrs, flux_obs, flux_ivar, mask,
                                    theta_lb, theta_ub),
                              threads=threads)  # **kwargs?

    # burn in
    pos0 = [theta0 + np.random.uniform(-1, 1, size=(len(theta0),)) * 1.e-3
            for _ in range(n_walkers)]
    pos, prob, rstate = sampler.run_mcmc(pos0, n_burnin)

    # run mcmc
    for i_run in range(mcmc_run_max_iter):
        print("--------------------------------------------------------------")
        print(prompt, " i_run : ", i_run)
        print(prompt, " Current pos : \n", pos)

        # new position
        pos_new, state, pos_best = check_chains(
            sampler, pos, theta_lb, theta_ub, mode_list=['bounds'])
        print(prompt, " New pos : ", pos_new)
        print(prompt, " Best pos : ", pos_best)

        if np.any(np.logical_not(state)):
            print(prompt, " Chain states : ", state)
            print(
            prompt, " RESET chain : ", np.arange(0, len(state) + 1)[state])

        # maximum correlation coefficients
        mcc_qtl, mcc_mat = sampler_mcc(sampler)
        # state_mcc = True --> not any out of threshold --> good chain
        state_mcc = ~np.any(np.abs(mcc_qtl) >= mcc)

        print(prompt, " *** MCC quantiles *** : ", mcc_qtl)
        # print(prompt, " MCC_MAT : -----------------------------------------")
        # for i in range(mcc_mat.shape[2]):
        #     print(prompt, " MCC_MAT[:,:,%s]: " % i, mcc_mat[:, :, i])

        # if chains are good, break and do statistics
        if state_mcc and i_run > 0:
            break

        # else continue running
        sampler.reset()
        pos, prob, rstate = sampler.run_mcmc(pos_new, n_run)

    print(prompt, ' state_mcc : ', state_mcc)

    # estimate percentiles
    theta_est_mcmc = np.nanpercentile(sampler.flatchain, [15., 50., 85.], axis=0)

    # format of theta_est_mcmc:
    # array([theta_p15,
    #        theta_p50,
    #        theta_p85])
    # e.g.:
    # array([[ 3.21908185,  5.66655696,  8.99618546],
    #        [ 3.22411158,  5.68827311,  9.08791289],
    #        [ 3.22909087,  5.71157073,  9.17812294]])

    # sampler is not returned, for saving memory
    if return_chain:
        result = {'theta': theta_est_mcmc,
                  'state_mcc': state_mcc,
                  'mcc_qtl': mcc_qtl,
                  'mcc_mat': mcc_mat,
                  'i_run': i_run,
                  'flatchain': sampler.flatchain}
    else:
        result = {'theta': theta_est_mcmc,
                  'state_mcc': state_mcc,
                  'mcc_qtl': mcc_qtl,
                  'mcc_mat': mcc_mat,
                  'i_run': i_run}

    return result
    # if not return_chain:
    #     return theta_est_mcmc
    # else:
    #     return theta_est_mcmc, sampler.flatchain


def theta_between(theta, theta_lb, theta_ub):
    """ check if theta is between [theta_lb, theta_ub] """
    state = np.all(theta.flatten() >= theta_lb.flatten()) and \
            np.all(theta.flatten() <= theta_ub.flatten()) #and \
            # np.all(np.isfinite())
    return state


def check_chains(sampler, pos, theta_lb, theta_ub,
                 mode_list=['bounds']):
    """ check chains

    1> reset out-of-bound chains
    2> reset all chains to max likelihood neighbours
    """
    mode_all = ['bounds', 'reset_all']

    for mode in mode_list:
        assert mode in mode_all

    n_walkers, n_step, n_dim = sampler.chain.shape

    # state of each chain
    state = np.ones((n_walkers,), dtype=np.bool)

    # the best position
    pos_best = sampler.flatchain[np.argsort(sampler.flatlnprobability)[-1]]

    # 'bounds' : chain pos should be between theta_lb, theta_ub
    if 'bounds' in mode_list:
        state = np.logical_and(state, np.array(
            [theta_between(pos[i], theta_lb, theta_ub) for i in
             range(n_walkers)]))

    # 'reset_all' : reset all chains
    if 'reset_all' in mode_list:
        state = np.logical_and(state,
                               np.zeros((n_walkers,), dtype=np.bool))

    # determine new pos
    pos_new = []
    for i, state_ in enumerate(state):
        if not state_:
            # state_ = False, reset
            pos_new.append(pos_best +
                           np.random.uniform(-1, 1,
                                             size=pos_best.shape) * 1.e-3)
        else:
            pos_new.append(pos[i])

    return np.array(pos_new), state, pos_best


# IMPORTANT : this function is designed to implement "adaptive burn in length"
def sampler_mcc(sampler):
    """ calculate correlation coefficient matrix of chains

    Parameters
    ----------
    sampler : emcee.EnsembleSampler instance
        sampler

    Returns
    -------
    mcc_qtl : ndarray [3,]
        the [25, 50, 75] th percentiles of coefs
    coefs : ndarray [n_chain, n_chain, n_dim]
        the corrcoef between each pair of chains

    """
    n_chain = sampler.k

    # correlation coefficient matrix
    coefs = chain_corrcoef(sampler)
    # set diagonal to np.nan
    for idim in range(coefs.shape[2]):
        for ichain in range(n_chain):
            coefs[ichain, ichain, idim] = np.nan

    # correlation coefficient quantile
    mcc_qtl = np.nanpercentile(coefs, [25., 50., 75.])

    # return quantiles
    return mcc_qtl, coefs


def chain_corrcoef(sampler):
    """ calculate correlation coefficients of chains

    Parameters
    ----------
    sampler: emcee.EnsembleSampler
        MCMC flatchain

    Returns
    -------
    coefs : ndarray [n_chain, n_chain, n_dim]
        the corrcoef between each pair of chains

    """
    n_chain = sampler.k
    n_dim = sampler.dim

    coefs = np.zeros((n_chain, n_chain, n_dim))
    for i in range(n_chain):
        for j in range(n_chain):
            for k in range(n_dim):
                coefs[i, j, k] = np.corrcoef(sampler.chain[i, :, k],
                                             sampler.chain[j, :, k])[1, 0]
    return coefs


# deprecated
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


# deprecated
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


# deprecated
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

    coefs = chain_corrcoef(fchain, n_step)

    return np.mean(coefs) - 1. / n_chain