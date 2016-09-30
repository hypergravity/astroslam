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


__all__ = ['compare_labels', 'compare_spectra']


# TODO: compare tr_labels and test_labels / any two sets of labels
def compare_labels(label1, label2, labelname1='Label1', labelname2='Label2',
                   figsize=None, figpath=None, ):
    """ compare two sets of labels

    Parameters
    ----------
    label1 : ndarray (n_obs, n_dim)
        label set 1
    label2 : ndarray (n_obs, n_dim)
        label set 2
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
        figsize = (3. * n_dim, 6)

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
        ax.set_ylabel('%s : %s - %s : %s' % (labelname1, i, labelname2, i))

    fig.tight_layout()

    if figpath is not None:
        fig.savefig(figpath)

    return fig


# TODO: compare spectra!!! urgently!!!
def compare_spectra(spectra1, spectra2, ofst_step=0.2, plt_max=100):
    pass
