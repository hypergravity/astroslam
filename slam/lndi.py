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
- Thu Nov 17 19:00:00 2016

Modifications
-------------
- Thu Nov 17 19:00:00 2016

Aims
----
- Lndi class:
    An interpolator making use of scipy.interpolate.LinearNDInterpolator.
    This is particularly designed for synthetic spectra.

"""

from __future__ import print_function

import os

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from astropy.table import Table
from joblib import load, dump, Parallel, delayed

from .normalization import normalize_spectra_block
from .hyperparameter import summarize_hyperparameters_to_table, summarize_table
from .predict import predict_labels, predict_labels_chi2, predict_spectrum
from .standardization import standardize, standardize_ivar
from .train import train_multi_pixels, train_single_pixel
from .mcmc import predict_label_mcmc
from .diagnostic import compare_labels, single_pixel_diagnostic

__all__ = ['Lndi']


class Lndi(object):
    """
    An interpolator implemented based on scipy.interpolate.LinearNDInterpolator
    """
    wave = None
    tr_flux = None
    tr_label = None
    lndi = None
    trained = False

    def __init__(self, wave, tr_flux, tr_label):
        """ Constructor of an Lndi instance """
        self.wave = wave
        self.tr_label = tr_label
        self.tr_flux = tr_flux

        self.lndi = LinearNDInterpolator(tr_label, tr_flux)
        self.trained = True

        self.ntrain = tr_label.shape[0]
        self.ndim = tr_label.shape[1]

    def predict_spectra(self, test_label):
        """ predict spectra given labels """
        return self.lndi(test_label)

    def generate_spectra_rand(self, X_init, X_std, n_rand):
        """ generate spectra randomly

        Parameters
        ----------
        X_init: ndarray (1, ndim)
            the central X
        X_std: ndarray (1, ndim)
            the std of gaussian random numbers
        n_rand: int
            the central X

        """
        X_rand = np.random.randn(n_rand, self.ndim)
        X_rand = X_init + X_rand * X_std.reshape(1, self.ndim)
        return self.predict_spectra(X_rand)

    def generate_label_rand(self, X_init, X_std, n_rand):
        """ generate labels randomly

        Parameters
        ----------
        X_init: ndarray (1, ndim)
            the central X
        X_std: ndarray (1, ndim)
            the std of gaussian random numbers
        n_rand: int
            the central X

        """
        X_rand = np.random.randn(n_rand, self.ndim)
        X_rand = X_init + X_rand * X_std.reshape(1, self.ndim)
        return X_rand

    def predict_label_rand(self, flux_goal, X_init, X_std,
                           n_rand, frac_ext=4, n_0_th=2,
                           verbose=False):
        n_loop = 0
        n_0 = 0
        while True:
            n_loop += 1

            if verbose:
                print('@Cham: n_loop = %s ...' % n_loop)

            # generate random numbers
            n_rand = np.int(n_rand)
            frac_ext = np.int(frac_ext)
            n_rand_2 = n_rand / frac_ext
            n_rand_1 = n_rand - n_rand_2
            X_rand_1 = self.generate_label_rand(X_init, X_std, n_rand_1)
            X_rand_2 = self.generate_label_rand(X_init, X_std*3., n_rand_2)
            X_rand = np.vstack((X_init, X_rand_1, X_rand_2))
            flux_rand = self.predict_spectra(X_rand)

            # kick nan
            ind_nan = np.any(np.isnan(flux_rand), axis=1)
            X_rand = X_rand[~ind_nan]
            flux_rand = flux_rand[~ind_nan]
            # find chi2_min
            i_min, chi2_min, flux_cont_min = best_chi2(
                self.wave, flux_rand, flux_goal)

            if verbose:
                print('@Cham: ', i_min, chi2_min)

            # if n_0 > n_0_th, but stuck in the same place --> end
            if i_min == 0:
                n_0 += 1
                if n_0 > n_0_th:
                    break
            else:
                n_0 = 0
                X_init = X_rand[i_min]

        return(X_rand[i_min], chi2_min,
               flux_cont_min, flux_rand[i_min]*flux_cont_min)


def best_chi2(wave, flux, flux_goal, ivar=None):
    flux_cont = determine_continuum(wave, flux, flux_goal,
                                    norm_range=(4000, 8000),
                                    dwave=100, p=(1E-7, 1E-7))
    if ivar is None:
        chi2 = np.nansum((flux*flux_cont-flux_goal)**2., axis=1)
    else:
        chi2 = np.nansum((flux*flux_cont - flux_goal) ** 2. *
                         ivar.reshape(1, -1), axis=1)

    i_min = np.argmin(chi2)
    chi2_min = chi2[i_min]
    return i_min, chi2_min, flux_cont[i_min]


def determine_continuum(wave, flux, flux_goal,
                        norm_range=(4000, 8000), dwave=100, p=(1E-7, 1E-7),
                        q=.5, rsv_frac=3):
    """ determine the best continuum making flux to fit flux_goal """
    # TODO: ivar should be considered for emission line stars
    flux_cont = normalize_spectra_block(wave, flux_goal/flux,
                                        norm_range=norm_range, dwave=dwave,
                                        p=p, q=q, ivar_block=None,
                                        rsv_frac=rsv_frac,
                                        n_jobs=1, verbose=False)[1]
    return flux_cont
