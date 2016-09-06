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
- normalization

"""
from __future__ import division

import numpy as np
from joblib import Parallel, delayed

from .extern.interpolate import SmoothSpline


# TODO: should take into account ivar?
def normalize_spectrum(wave, flux, norm_range, dwave, p=(1E-6, 1E-6), q=0.5):
    """ A double smooth normalization of a spectrum
    Converted from Chao Liu's normSpectrum.m

    Parameters
    ----------
    wave: ndarray (n_pix, )
        wavelegnth array
    flux: ndarray (n_pix, )
        flux array
    norm_range: tuple
        a tuple consisting (wave_start, wave_stop)
    dwave: float
        binning width
    p: tuple of 2 ps
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    q: float in range of [0, 100]
        percentile, between 0 and 1

    Returns
    -------
    flux_norm: ndarray
        normalized flux
    flux_cont: ndarray
        continuum flux

    Example
    -------
    >>> flux_norm, flux_cont = normalize_spectrum(
    >>>     wave, flux, (4000., 8000.), 100., p=(1E-8, 1E-7), q=0.5)

    """

    assert 0. < q < 1.

    # n_iter = len(p)
    n_bin = np.fix(np.diff(norm_range) / dwave) + 1
    wave1 = norm_range[0]

    # SMOOTH 1
    flux_smoothed1 = SmoothSpline(wave, flux, p[0])(wave)
    dflux = flux - flux_smoothed1

    # collecting continuum pixels --> ITERATION 1
    ind_good = np.zeros(wave.shape, dtype=np.bool)
    for i_bin in range(n_bin):
        ind_bin = np.logical_and(wave > wave1 + (i_bin - 0.5) * dwave,
                                 wave <= wave1 + (i_bin + 0.5) * dwave)
        if np.sum(ind_bin > 0):
            # median & sigma
            bin_median = np.median(dflux[ind_bin])
            bin_std = np.median(np.abs(dflux - bin_median))
            # within 1 sigma with q-percentile
            ind_good = np.logical_or(ind_good, (np.abs(
                dflux - np.percentile(dflux[ind_bin], q * 100.)) < (
                                                    1. * bin_std)) * ind_bin)

    # assert there is continuum pixels
    try:
        assert np.sum(ind_good) > 0
    except AssertionError:
        Warning("@Keenan.normalize_spectrum(): unable to find continuum! ")
        ind_good = np.ones(wave.shape, dtype=np.bool)

    # SMOOTH 2
    # continuum flux
    flux_smoothed2 = SmoothSpline(wave[ind_good], flux[ind_good], p[1])(wave)
    # normalized flux
    flux_norm = flux / flux_smoothed2

    return flux_norm, flux_smoothed2


def normalize_spectra_block(wave, flux_block, norm_range, dwave,
                            p=(1E-6, 1E-6), q=0.5, n_jobs=1, verbose=10):
    """ normalize multiple spectra using the same configuration
    This is specially designed for TheKeenan

    Parameters
    ----------
    wave: ndarray (n_pix, )
        wavelegnth array
    flux_block: ndarray (n_obs, n_pix)
        flux array
    norm_range: tuple
        a tuple consisting (wave_start, wave_stop)
    dwave: float
        binning width
    p: tuple of 2 ps
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    q: float in range of [0, 100]
        percentile, between 0 and 1
    n_jobs: int
        number of processes launched by joblib
    verbose: int / bool
        verbose level

    Returns
    -------
    flux_norm: ndarray
        normalized flux

    """

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(normalize_spectrum)(
            wave, flux_item, norm_range, dwave, p=p, q=q)
        for flux_item in flux_block)

    # unpack results
    flux_norm_block = []
    flux_cont_block = []
    for result in results:
        flux_norm_block.append(result[0])
        flux_cont_block.append(result[1])

    return flux_norm_block, flux_cont_block


def normalize_spectra(wave_flux_tuple_list, norm_range, dwave,
                      p=(1E-6, 1E-6), q=50, n_jobs=1, verbose=False):
    """ normalize multiple spectra using the same configuration

    Parameters
    ----------
    wave_flux_tuple_list: list[n_obs]
        a list of (wave, flux) tuple
    norm_range: tuple
        a tuple consisting (wave_start, wave_stop)
    dwave: float
        binning width
    p: tuple of 2 ps
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    q: float in range of [0, 100]
        percentile, between 0 and 1
    n_jobs: int
        number of processes launched by joblib
    verbose: int / bool
        verbose level

    Returns
    -------
    flux_norm: ndarray
        normalized flux


    """
    # TODO: this is a generalized version
    pass
