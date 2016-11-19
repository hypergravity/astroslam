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
- Tue Oct 04 13:00:00 2016

Modifications
-------------
- Tue Oct 04 13:00:00 2016

Aims
----
- to implement functions for binning pixels

"""

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d


def interp_pchip(wave, spec, wave_interp, extrapolate=False):
    """ interpolate for single spectrum (pchip)

    Parameters
    ----------
    wave: ndarray
        wavelength array
    spec: ndarray
        spectrum array
    wave_interp: ndarray
        wavelength array to be interpolated to
    extrapolate: bool
        if True, extrapolate
        if False, return NaNs for out-of-bounds pixels

    Returns
    -------
    spec_interp: ndarray
        interpolated spectrum

    """
    P = PchipInterpolator(wave, spec, extrapolate=extrapolate)
    spec_interp = P(wave_interp)
    return spec_interp


def interp_linear(wave, spec, wave_interp, fill_value=np.nan):
    """ interpolate for single spectrum (linear)

    Parameters
    ----------
    wave: ndarray
        wavelength array
    spec: ndarray
        spectrum array
    wave_interp: ndarray
        wavelength array to be interpolated to
    fill_value: float/nan
        fill out-of-bounds pixels with fill_value

    Returns
    -------
    spec_interp: ndarray
        interpolated spectrum

    """
    I = interp1d(wave, spec, kind='linear',
                 bounds_error=False, fill_value=fill_value)
    return I(wave_interp)


def interp_cubic(wave, spec, wave_interp, fill_value=np.nan):
    """ interpolate for single spectrum (cubic)

    Parameters
    ----------
    wave: ndarray
        wavelength array
    spec: ndarray
        spectrum array
    wave_interp: ndarray
        wavelength array to be interpolated to
    fill_value: float/nan
        fill out-of-bounds pixels with fill_value

    Returns
    -------
    spec_interp: ndarray
        interpolated spectrum

    """
    I = interp1d(wave, spec, kind='cubic',
                 bounds_error=False, fill_value=fill_value)
    return I(wave_interp)


def interp_nearest(wave, spec, wave_interp, fill_value=np.nan):
    """ interpolate for single spectrum (nearest)

    Parameters
    ----------
    wave: ndarray
        wavelength array
    spec: ndarray
        spectrum array
    wave_interp: ndarray
        wavelength array to be interpolated to
    fill_value: float/nan
        fill out-of-bounds pixels with fill_value

    Returns
    -------
    spec_interp: ndarray
        interpolated spectrum

    """
    I = interp1d(wave, spec, kind='nearest',
                 bounds_error=False, fill_value=fill_value)
    return I(wave_interp)


def add_noise_normal(flux, snr):
    """ add normal random noise for flux (single spectrum)

    Parameters
    ----------
    flux: ndarray
        flux array
    snr: float
        Signal-to-Noise Ratio

    Returns
    -------
    flux: ndarray

    """
    nsr = np.random.randn(*flux.shape) / snr
    nsr = np.where((nsr < 1.) * (nsr > -1.), nsr, np.zeros_like(flux))

    return flux * (1. + nsr)


def add_noise_gpoisson(flux, k=1.0):
    """ add SCALED Poisson random noise for flux (single spectrum)

    Parameters
    ----------
    flux: ndarray
        flux array
    k: float
        k times better Poisson noise, implemented in case Poisson is too noisy
        default value is 1.

    Returns
    -------
    flux: ndarray

    """
    nsr = np.random.randn(*flux.shape) / np.sqrt(np.abs(flux)) / k
    nsr = np.where((nsr < 1.) * (nsr > -1.), nsr, np.zeros_like(flux))

    return flux * (1. + nsr)


def add_noise_poisson(flux):
    """ add Poisson random noise for flux (single/multi spectrum)

    Parameters
    ----------
    flux: ndarray
        flux array

    Returns
    -------
    flux: ndarray

    """
    return np.random.poisson(flux)


def measure_poisson_snr(flux):
    """ measure Poisson SNR  for flux

    Parameters
    ----------
    flux: ndarray 2D
        flux

    Returns
    -------
    snr_med: ndarray
        the median Poisson SNR of flux

    """
    # Poisson SNR
    snr = np.sqrt(flux)
    # median Poisson SNR
    snr_med = np.median(snr, axis=1)

    return snr_med


def shift_poisson_snr(flux, snr):
    """ shift Poisson SNR for flux

    Parameters
    ----------
    flux: ndarray 1D/2D
        flux
    snr: float
        target snr

    Returns
    -------
    flux__ : ndarray 2D
        flux with median SNR = snr

    """

    if flux.ndim == 1:
        # 1d flux
        flux = flux.reshape(1, -1)
    elif flux.ndim > 2:
        # >2d
        raise(ValueError('The number of dimensions of input flux is larger than 2!'))

    # measure poisson SNR for flux
    snr_med = measure_poisson_snr(flux)[:, None]
    # determine scale
    scale_ = (snr_med/snr) ** 2.
    # scale flux
    flux_ = flux / scale_

    if flux.ndim == 1:
        flux_ = flux_.flatten()

    return flux_


def binning_pixels(wave, flux, ivar=None, n_pixel=3):
    """

    Parameters
    ----------
    wave: ndarray
        wavelength array
    flux: ndarray
        flux array
    ivar: ndarray
        ivar array
    n_pixel: int
        number of pixels binned

    Returns
    -------
    binned_wave: ndarray
        binned wavelength array
    binned_flux:
        binned flux array
    binned_ivar:
        binned ivar array

    """
    assert n_pixel > 0

    # default ivar
    if ivar is None:
        ivar = np.ones_like(flux)

    # determine the number of binned pixels
    n_binned = np.fix(len(flux) / n_pixel)

    # initialization
    binned_wave = np.ones(n_binned)
    binned_flux = np.ones(n_binned)
    binned_ivar = np.ones(n_binned)

    # iterate for each binned pixel [wave, flux, ivar]
    for i_pix in range(n_binned):
        binned_wave[i_pix] = np.mean(
            wave[i_pix * n_pixel:(i_pix + 1) * n_pixel])
        binned_flux[i_pix] = np.mean(
            flux[i_pix * n_pixel:(i_pix + 1) * n_pixel])
        this_ivar_array = ivar[i_pix * n_pixel:(i_pix + 1) * n_pixel]
        if np.all((this_ivar_array > 0.) * np.isfinite(this_ivar_array)):
            # all pixels are good
            # ################## binning method #################### #
            # (err1**2 + err2**2 + ... + errn**2) / n**2 = errbin**2 #
            # 1/ivar1 + 1/ivar2 + ... + 1/ivarn = n**2 /ivar         #
            # --> binning n pixels with the same error               #
            # --> improves SNR by a factor of sqrt(n)                #
            # ###################################################### #
            binned_ivar[i_pix] = n_pixel ** 2. / np.sum(1. / this_ivar_array)
        else:
            # bad pixel exists
            binned_ivar[i_pix] = 0.

    return binned_wave, binned_flux, binned_ivar


def test_interpolation():
    x = np.arange(0., 10., 1.)
    y = np.sin(x)
    plt.plot(x, y, 'r')
    xx = np.arange(0., 10., 0.2)
    plt.plot(xx, interp_pchip(x, y, xx), 'b')
    plt.plot(xx, interp_linear(x, y, xx), 'g')
    plt.plot(xx, interp_cubic(x, y, xx), 'c')
    plt.plot(xx, interp_nearest(x, y, xx), 'm')


if __name__ == "__main__":
    test_interpolation()
