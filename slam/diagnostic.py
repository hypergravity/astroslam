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
- Tue Sep 06 21:00:00 2016

Modifications
-------------
- Tue Sep 06 21:00:00 2016

Aims
----
- diagnostics of tr/test_labels
- diagnostics of fitted spectra

"""

import numpy as np
import matplotlib.pyplot as plt

from .predict import predict_pixel_for_diagnostic


__all__ = ['compare_labels', 'compare_spectra']


def compare_labels(label1, label2, labelname1='Label1', labelname2='Label2',
                   figsize=None, figpath=None, ):
    """ compare two sets of labels

    Parameters
    ----------
    label1 : ndarray (n_obs, n_dim)
        label set 1 (X)
    label2 : ndarray (n_obs, n_dim)
        label set 2 (Y)
    labelname1: string
        name of label1
    labelname2: string
        name of label2
    figsize : tuple of float
        figure size
    figpath : string
        filepath of this figure

    """

    assert label1.shape == label2.shape
    n_obs, n_dim = label1.shape

    # default figsize
    if figsize is None:
        figsize = (3.2 * n_dim, 6)

    # draw figure
    fig = plt.figure(figsize=figsize)
    for i in range(n_dim):
        x, y = label1[:, i], label2[:, i]
        xy = np.stack([x, y])
        xlim = (np.min(xy), np.max(xy))

        # diagnal plot
        ax = fig.add_subplot(2, n_dim, i + 1)
        ax.plot(x, y, 'b.')
        ax.plot(xlim, xlim, 'k--')
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_xlabel('%s : %s' % (labelname1, i))
        ax.set_ylabel('%s : %s' % (labelname2, i))

        # diff plot
        ax = fig.add_subplot(2, n_dim, i + 1 + n_dim)
        ax.plot(x, y - x, 'b.')
        ax.plot(xlim, [0., 0.], 'k--')
        ax.set_xlim(xlim)
        ax.set_xlabel('%s : %s' % (labelname1, i))
        ax.set_ylabel('%s : %s - %s : %s' % (labelname2, i, labelname1, i))

    fig.tight_layout()

    if figpath is not None:
        fig.savefig(figpath)

    return fig


def compare_spectra(spectra1, spectra2=None, ofst_step=0.2, wave=None,
                    mediannorm=False, figsize=(10, 6), plt_max=100):
    """ compare one/two spectra set """
    n_spec = spectra1.shape[0]

    # if mediannorm is a float, scale spectra to median*
    if isinstance(mediannorm, float):
        for i in range(n_spec):
            spectra1[i] /= np.nanmedian(spectra1[i]) * mediannorm
        if spectra2 is not None:
            for i in range(n_spec):
                spectra2[i] /= np.nanmedian(spectra2[i]) * mediannorm

    # plot the figure
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(111)
    if wave is None:
        for i in range(n_spec):
            ofst = i * ofst_step
            plt.plot(spectra1[i] + ofst, 'b')
            if spectra2 is not None:
                plt.plot(spectra2[i] + ofst, 'r')
    else:
        for i in range(n_spec):
            ofst = i * ofst_step
            plt.plot(wave, spectra1[i] + ofst, 'b')
            if spectra2 is not None:
                plt.plot(wave, spectra2[i] + ofst, 'r')

    return fig


def single_pixel_diagnostic(svrs,
                            i_pixel,
                            test_labels,
                            diag_dim=(0,),
                            labels_scaler=None,
                            flux_scaler=None):
    """ diagnostic a single pixel in 1D/2D

    Parameters
    ----------
    svrs: list of sklearn.smv.SVR instance
        k.svrs
    i_pixel: int
        No. of pixel that will be in diagnostic
    test_labels: ndarray ( n, ndim )
        test labels
    diag_dim: tuple/list
        diagnostic dimensions, e.g., (0, 1)
    labels_scaler:
        scaler for labels, e.g., k.tr_labels_scaler
    flux_scaler:
        scaler for flux, e.g., k.tr_flux_scaler

    Returns
    -------
    [X, (Y,) flux]

    """
    # assertions
    assert 1 <= len(diag_dim) <= 2
    for dim_ in diag_dim:
        assert 0 <= dim_ <= test_labels.shape[1]

    # draw scaling parameters for this pixel
    if flux_scaler is None:
        flux_mean_ = 0.
        flux_scale_ = 1.
    else:
        flux_mean_ = flux_scaler.mean_[i_pixel]
        flux_scale_ = flux_scaler.scale_[i_pixel]

    # prdict flux for this pixel
    pixel_flux = predict_pixel_for_diagnostic(
        svrs[i_pixel], test_labels,
        labels_scaler=labels_scaler,
        flux_mean_=flux_mean_,
        flux_scale_=flux_scale_)

    result = []
    for dim_ in diag_dim:
        result.append(test_labels[:, dim_])
    result.append(pixel_flux.flatten())

    # return [X, (Y,) flux]
    return result
