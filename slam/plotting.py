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
- Sat Sep 03 16:00:00 2017

Modifications
-------------
- Sat Sep 03 12:00:00 2017

Aims
----
- plotting tools

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib import cm
from lmfit.models import GaussianModel
from mpl_toolkits.mplot3d import Axes3D

from .analysis import label_diff_lmfit


def plot_mse(s):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
    plt.hist(-s.nmse[s.nmse != 0], np.linspace(0, 1, 80), histtype='step',
             lw=2, label="MSE")
    plt.hist(-s.scores[s.nmse != 0], np.linspace(0, 1, 80), histtype='step',
             lw=2, label="CV MSE")
    ylim = plt.gca().get_ylim()
    plt.vlines(np.percentile(-s.nmse[s.nmse != 0], [14, 50, 86]), *ylim,
               linestyle='--', label="14, 50, 86 percentiles")
    plt.xlim(0, 1)
    plt.ylim(*ylim)
    plt.ylabel("Counts")
    plt.xlabel("MSE")
    fig.tight_layout()
    return fig


# ################ #
# image
# ################ #
def image(ax, x, y, xbins, ybins, log=True):
    plt.sca(ax)
    c, xe, ye, bn = binned_statistic_2d(x, y, x, statistic="count",
                                        bins=[xbins, ybins])
    if log:
        c = np.log10(c)
    plt.imshow(c.T, origin="lower", extent=(*xbins[[0, -1]], *ybins[[0, -1]]),
               cmap=cm.viridis, aspect="auto")
    return


def compare_labels(X_true, X_pred,
                   xlabels=None, ylabels=None, reslabels=None,
                   xlims=None, reslims=None,
                   histlim=None, nxb=30, cornerlabel="",
                   figsize=None):

    nlabel = X_true.shape[1]

    if xlabels is None:
        xlabels = ["$X_{{true}}:{}$".format(i) for i in range(nlabel)]
    if ylabels is None:
        ylabels = ["$X_{{pred}}:{}$".format(i) for i in range(nlabel)]
    if reslabels is None:
        reslabels = ["$X_{{res}}:{}$".format(i) for i in range(nlabel)]

    # default xlim
    if xlims is None:
        xlim1 = np.min(np.vstack((np.percentile(X_true, 1, axis=0),
                                  np.percentile(X_pred, 1, axis=0))), axis=0)
        xlim2 = np.min(np.vstack((np.percentile(X_true, 99, axis=0),
                                  np.percentile(X_pred, 99, axis=0))), axis=0)
        xlims = (xlim2 - xlim1).reshape(-1, 1) * 0.4 * np.array(
            [-1, 1]) + np.vstack((xlim1, xlim2)).T
    if reslims is None:
        reslims = np.repeat(
            np.max(np.abs(np.percentile(X_pred - X_true, [1, 99], axis=0).T),
                   axis=1).reshape(-1, 1), 2, axis=1) * np.array([-1, 1])
        reslims = np.abs(np.diff(reslims, axis=1)) * np.array(
            [-1, 1]) * 0.2 + reslims

    # run MCMC
    X_bias, X_scatter, frs, histdata = label_diff_lmfit(
        X_true, X_pred, bins="auto", plot=False, emcee=True)
    print("bias", X_bias)
    print("scatter", X_scatter)
    if histlim is None:
        histlim = (0, np.max([np.max(histdata_[0]) for histdata_ in histdata]))
    histlim = np.array(histlim)

    if figsize is None:
        figsize = (3 * nlabel, 3 * nlabel)

    # draw figure
    fig, axs2 = plt.subplots(nlabel+1, nlabel+1, figsize=figsize)

    # 1. Gaussian
    gm = GaussianModel()
    for i in range(nlabel):
        plt.sca(axs2[i + 1, -1])
        fr = frs[i]
        hist_, bin_edge_, data_ = histdata[i]
        plt.hist(data_, bins=bin_edge_, histtype="step",
                 orientation="horizontal")
        axs2[i + 1, -1].plot(gm.eval(fr.mcmc.params, x=bin_edge_), bin_edge_)
        axs2[i + 1, -1].tick_params(direction='in', pad=5)
        axs2[i + 1, -1].set_xlim(histlim)
        axs2[i + 1, -1].set_ylim(reslims[i])
        axs2[i + 1, -1].set_ylim(reslims[i])
        axs2[i + 1, -1].yaxis.tick_right()
        axs2[i + 1, -1].hlines(X_bias[i], *histlim, linestyle='--', color="k")

        pos_text_x = np.dot(np.array([[0.9, 0.1]]), histlim.reshape(-1, 1))
        pos_text_y = np.dot(np.array([[0.15, 0.85]]),
                            reslims[i].reshape(-1, 1))
        axs2[i + 1, -1].text(pos_text_x, pos_text_y,
                             "$bias={:.4f}$".format(X_bias[i]))
        pos_text_x = np.dot(np.array([[0.9, 0.1]]), histlim.reshape(-1, 1))
        pos_text_y = np.dot(np.array([[0.30, 0.70]]),
                            reslims[i].reshape(-1, 1))
        axs2[i + 1, -1].text(pos_text_x, pos_text_y,
                             "$\\sigma={:.4f}$".format(X_scatter[i]))

        axs2[i + 1, -1].yaxis.tick_right()

        if i < nlabel-1:
            axs2[i + 1, -1].set_xticklabels([])

    axs2[-1, -1].set_xlabel("Counts")

    # 2. diagnal
    for i in range(nlabel):
        image(axs2[0, i], X_true[:, i], X_pred[:, i],
              np.linspace(xlims[i][0], xlims[i][1], nxb),
              np.linspace(xlims[i][0], xlims[i][1], nxb))
        axs2[0, i].set_xlim(*xlims[i])
        axs2[0, i].set_ylim(*xlims[i])
        axs2[0, i].tick_params(direction='in', pad=5)
        axs2[0, i].set_xticklabels([])
        axs2[0, i].set_ylabel(ylabels[i])
        axs2[0, i].plot(xlims[i], xlims[i], 'k--')

    # 3. Xres vs X
    X_res = X_pred - X_true
    for i in range(nlabel):
        for j in range(nlabel):
            image(axs2[j + 1, i], X_true[:, i], X_res[:, j],
                  np.linspace(xlims[i][0], xlims[i][1], nxb),
                  np.linspace(reslims[j][0], reslims[j][1], nxb))
            axs2[j + 1, i].set_xlim(*xlims[i])
            axs2[j + 1, i].set_ylim(*reslims[j])
            axs2[j + 1, i].tick_params(direction='in', pad=5)

            if j != nlabel - 1:
                axs2[j + 1, i].set_xticklabels([])
            else:
                axs2[j + 1, i].set_xlabel(xlabels[i])

            if i != 0:
                axs2[j + 1, i].set_yticklabels([])
            else:
                axs2[j + 1, i].set_ylabel(reslabels[j])

    axs2[0, -1].set_axis_off()
    axs2[0, -1].text(np.mean(axs2[0, -1].get_xlim()),
                     np.mean(axs2[0, -1].get_ylim()),
                     cornerlabel, horizontalalignment='center',
                     verticalalignment='center')

    fig.tight_layout()
    plt.subplots_adjust(wspace=0., hspace=0.)

    return fig, frs

# gs1 = plt.GridSpec(1, 6, left=0.05, bottom=0.85, right=0.95, top=0.95, hspace=0.2, wspace=0)
# gs2 = plt.GridSpec(7, 7, left=0.05, bottom=0.1, right=0.95, top=0.8, hspace=0, wspace=0)

# axs1 = np.array([fig.add_subplot(gs1[i]) for i in range(6)])
# axs2 = np.array([[fig.add_subplot(gs2[j, i]) for i in range(7)] for j in range(7)])


def visual3d(s, wave, diag_dim=()):
    """ single pixel diagnostic """

    diag_dim = (0, 1)

    i_pixel = 5175 - 3900
    # i_pixel = 4861-3900
    x, y, flux = s.single_pixel_diagnostic(i_pixel, s.tr_labels,
                                           diag_dim=diag_dim)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(s.tr_labels[:, 0], s.tr_labels[:, 1], s.tr_flux[:, i_pixel],
                 s=10, c=s.tr_labels[:, 2], alpha=.5, cmap=cm.jet)
    plt.colorbar()
    # plt.plot(x,y, flux,'b.')
    ax.set_zlim(0., 2.)
    plt.xlabel('Teff')
    plt.ylabel('logg')
    plt.title('PIXEL: %s' % i_pixel)
    fig.tight_layout()

    # %%
    diag_dim = (0, 2)

    i_pixel = 6564 - 3900
    i_pixel = 4861 - 3900

    sdiag_teff = np.arange(4000., 8000., 100.)
    sdiag_logg = np.arange(1., 5., .2)
    sdiag_logg = np.arange(-2, 1., .2)
    msdiag_teff, msdiag_logg = np.meshgrid(sdiag_teff, sdiag_logg)
    msdiag_feh = np.zeros_like(msdiag_teff)
    msdiag_labels = np.array([msdiag_teff.flatten(),
                              msdiag_logg.flatten(),
                              msdiag_feh.flatten()]).T

    H, xedges, yedges = np.histogram2d(train_labels[:, 0], train_labels[:, 2],
                                       bins=(sdiag_teff, sdiag_logg))
    xcenters = (xedges[:-1] + xedges[1:]) / 2.
    ycenters = (yedges[:-1] + yedges[1:]) / 2.
    xmesh, ymesh = np.meshgrid(xcenters, ycenters)

    x, y, flux = s.single_pixel_diagnostic(i_pixel, msdiag_labels,
                                           diag_dim=diag_dim)

    # flux[:20]=np.nan

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x,y,flux, c=msdiag_feh)
    surf = ax.plot_surface(msdiag_teff, msdiag_logg,
                           flux.reshape(msdiag_teff.shape),
                           vmin=np.min(flux), vmax=np.max(flux), cmap=cm.jet)
    ax.contour(xmesh, ymesh, np.log(H.T), extend3d=False, offset=1.20,
               color='k')
    # plt.plot(x,y, flux,'b.')
    ax.set_zlim(0., 2.)
    plt.xlabel('Teff')
    plt.ylabel('logg')
    plt.title('PIXEL: %s' % i_pixel)

    fig.colorbar(surf, shrink=.5, aspect=5)
    fig.tight_layout()
    # fig.savefig("../data/laap/figs/PIXEL6564_C8_E0P08_G1.svg")