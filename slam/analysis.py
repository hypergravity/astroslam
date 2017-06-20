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
- Tue Jun 20 10:00:00 2017

Modifications
-------------
- Tue Jun 20 10:00:00 2017

Aims
----
- error analysis

"""

import numpy as np
from scipy.optimize import minimize, least_squares, curve_fit
from matplotlib import pyplot as plt
from lmfit.models import GaussianModel


# ################ #
# likelihood fit
# ################ #

def lnprior(theta):
    mu, sigma = theta
    if sigma < 0:
        return -np.inf
    else:
        return 0


def lnlike(theta, data):
    mu, sigma = theta
    return np.sum(-(data - mu) ** 2 / sigma ** 2 / 2) - \
           len(data) * np.log(sigma)


def lnpost(theta, data):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return lp
    else:
        return lnlike(theta, data) + lp


def nlnpost(*args):
    return -lnpost(*args)


def gfit_mle(theta0, data):
    return minimize(nlnpost, theta0, args=(data))


def label_diff_mle(label1, label2):
    label1 = np.array(label1)
    label2 = np.array(label2)
    assert label1.shape == label2.shape

    n_obs, n_dim = label1.shape
    bias = np.zeros((n_dim,), dtype=float)
    scatter = np.zeros((n_dim,), dtype=float)

    for i_dim in range(n_dim):
        data = label1[:, i_dim] - label2[:, i_dim]
        theta0 = np.array([np.median(data), 2 * np.std(data)])
        x = gfit_mle(theta0, data)
        if x.success:
            bias[i_dim], scatter[i_dim] = x['x']
        else:
            Warning("@GFIT: not successful [i_dim={}]!".format(i_dim))
            print("------- X [i_dim={}] -------".format(i_dim))
            print(x)
            bias[i_dim], scatter[i_dim] = x['x']

    return bias, scatter


def test_gfit_mle():
    data = np.random.randn(10000, )
    theta0 = np.array([1., 1.])
    print(gfit_mle(theta0, data))


# ################ #
# binned fit
# ################ #

def gauss1d(x, a, b, c):
    return a / (np.sqrt(2. * np.pi) * c) * np.exp(
        -(x - b) ** 2 / c ** 2 / 2)


def gauss1d_cost(theta, x, y):
    a, b, c = theta
    if a <= 0 or c <= 0:
        return -np.inf
    return gauss1d(theta, x) - y


def gfit_bin(theta0, data):
    # hist, bin_edges = np.histogram(data, bins='auto')
    bins = np.arange(np.min(data), np.max(data), np.std(data)/3)
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # figure();
    # plot(bin_centers, hist)
    # return least_squares(gauss1d_cost, theta0, args=(bin_centers, hist))
    return curve_fit(gauss1d, bin_centers, hist, p0=theta0,
                     bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))


def test_gfit_bin():
    data = np.random.randn(10000, )
    theta0 = np.array([1., np.median(data), 2 * np.std(data)])
    theta = gfit_bin(theta0, data)[0]
    print(theta)

    # hist, bin_edges = np.histogram(data, bins='auto')
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # figure();
    # plot(bin_centers, hist)
    # plot(bin_centers, gauss1d(theta, bin_centers))


def label_diff_bin(label1, label2, plot=False):
    label1 = np.array(label1)
    label2 = np.array(label2)
    assert label1.shape == label2.shape

    n_obs, n_dim = label1.shape
    amp = np.zeros((n_dim,), dtype=float)
    bias = np.zeros((n_dim,), dtype=float)
    scatter = np.zeros((n_dim,), dtype=float)

    for i_dim in range(n_dim):
        data = label1[:, i_dim] - label2[:, i_dim]
        # data = data[np.logical_and(data>np.percentile(data, 0), data<np.percentile(data, 100))]
        theta0 = np.array([len(data), np.median(data), 1 * np.std(data)])
        popt, pcov = gfit_bin(theta0, data)
        amp[i_dim], bias[i_dim], scatter[i_dim] = popt

    if plot:
        fig = plt.figure(figsize=(3*n_dim, 4))
        for i_dim in range(n_dim):
            ax = fig.add_subplot(1, n_dim, i_dim+1)
            data = label1[:, i_dim] - label2[:, i_dim]
            # data = data[np.logical_and(data > np.percentile(data, 0),
            #                            data < np.percentile(data, 100))]
            bins = np.arange(np.min(data), np.max(data), np.std(data) / 2)
            hist, bin_edges = np.histogram(data, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.step(bin_edges, np.hstack((hist, 0)), where='post')
            ax.plot(bin_edges, gauss1d(bin_edges, amp[i_dim], bias[i_dim], scatter[i_dim]))

    return bias, scatter


# ############################### #
# binned gaussian fit using LMFIT
# ############################### #

def gfit_bin_lmfit(data, bins='', bin_std=3, plot=False):
    if bins == 'robust':
        bins = np.arange(np.min(data), np.max(data), np.std(data)/bin_std)

    # binned statistics
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # fit a gaussian model to data using LMFIT
    gm = GaussianModel()
    theta_guess = gm.guess(hist, x=bin_centers)
    fr = gm.fit(hist, theta_guess, x=bin_centers, method="least_squares")
    # fr.fit_report()

    return (fr.values['amplitude'], fr.values['center'], fr.values['sigma']),fr


def label_diff_lmfit(label1, label2, bins='robust', bin_std=3, plot=False):
    """ label difference between label2 and label1(truth)
    
    Parameters
    ----------
    label1:
        truth
    label2:
        guess
    bins: str
        "auto" is recommended
    bin_std:
        binwidth = std/bin_std
    plot: bool
        if True, plot figure

    Returns
    -------

    """
    label1 = np.array(label1)
    label2 = np.array(label2)
    assert label1.shape == label2.shape

    n_obs, n_dim = label1.shape
    amp = np.zeros((n_dim,), dtype=float)
    bias = np.zeros((n_dim,), dtype=float)
    scatter = np.zeros((n_dim,), dtype=float)
    frs = np.zeros((n_dim,), dtype=object)

    for i_dim in range(n_dim):
        data = label2[:, i_dim] - label1[:, i_dim]
        theta, frs[i_dim] = \
            gfit_bin_lmfit(data, bins=bins, bin_std=bin_std, plot=False)
        amp[i_dim], bias[i_dim], scatter[i_dim] = theta
            
    if plot:
        gm = GaussianModel()
        fig = plt.figure(figsize=(3*n_dim, 4))
        for i_dim in range(n_dim):
            ax = fig.add_subplot(1, n_dim, i_dim+1)
            data = label2[:, i_dim] - label1[:, i_dim]

            # binned statistics
            if bins == 'robust':
                bins = np.arange(
                    np.min(data), np.max(data), np.std(data) / bin_std)
            hist, bin_edges = np.histogram(data, bins=bins)
            # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # bin_xx = np.linspace(bin_edges[0], bin_edges[-1], 100)
            print(bin_edges)
            ax.hist(data, bins=bin_edges, histtype='step')
            ax.plot(bin_edges, gm.eval(frs[i_dim].params, x=bin_edges))

    return bias, scatter, frs

if __name__ == "__main__":
    test_gfit_mle()
    test_gfit_bin()
