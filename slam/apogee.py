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
- Sat Oct 31 12:00:00 2016

Modifications
-------------
- Sat Oct 31 12:00:00 2016

Aims
----
- APOGEE utils

"""

from __future__ import print_function

import os
import urllib
import urllib.request
from collections import OrderedDict

import numpy as np
from astropy.io import fits
from astropy.table import Table, Column

__all__ = ["apStar_url", "apStar_download", "mkdir_loop"]


def reconstruct_wcs_coord_from_fits_header(hdr, dim=1):
    """ reconstruct wcs coordinates (e.g., wavelength array) """
    # assert dim is not larger than limit
    assert dim <= hdr["NAXIS"]

    # get keywords
    crval = hdr["CRVAL%d" % dim]
    cdelt = hdr["CDELT%d" % dim]
    try:
        crpix = hdr["CRPIX%d" % dim]
    except KeyError:
        crpix = 1

    # length of the current dimension
    naxis_ = hdr["NAXIS%d" % dim]

    # reconstruct wcs coordinates
    coord = np.arange(1 - crpix, naxis_ + 1 - crpix) * cdelt + crval
    return coord


def apStar_read(fp, full=False, meta=False, verbose=False):
    """ read apStar fits file

    Parameters
    ----------
    fp: string
        file path
    full: bool
        if False, return a simple version of apStar spec.
        if True, return a full version of apStar spec.
    meta: bool
        if True, attach Primary HDU header as spec.meta (OrderedDict)
    verbose: bool:
        if True, verbose.

    Returns
    -------
    spec (astropy.table.Table instance)

    Notes
    -----
    The url of the doc for apStar files:
    https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/APSTAR_VERS/
    TELESCOPE/LOCATION_ID/apStar.html

    HDU0: master header with target information
    HDU1: spectra: combined and individual
    HDU2: error spectra
    HDU3: mask spectra
    HDU4: sky spectra
    HDU5: sky error spectra
    HDU6: telluric spectra
    HDU7: telluric error spectra
    HDU8: table with LSF coefficients
    HDU9: table with RV/binary information

    """
    # read apStar file
    hl = fits.open(fp)

    # construct Table instance
    if not full:
        # not full apStar info, [wave, flux, flux_err, mask] only
        spec = Table([
            Column(
                10. ** reconstruct_wcs_coord_from_fits_header(hl[1].header, 1),
                "wave"),
            Column(hl[1].data.T, "flux"),
            Column(hl[2].data.T, "flux_err"),
            Column(hl[3].data.T, "mask")])
    else:
        # full apStar info
        spec = Table([
            Column(
                10. ** reconstruct_wcs_coord_from_fits_header(hl[1].header, 1),
                "wave"),
            Column(hl[1].data.T, "flux"),
            Column(hl[2].data.T, "flux_err"),
            Column(hl[3].data.T, "mask"),
            Column(hl[4].data.T, "sky"),
            Column(hl[5].data.T, "sky_err"),
            Column(hl[6].data.T, "telluric"),
            Column(hl[7].data.T, "telluric_err")])

    # meta data
    if meta:
        spec.meta = OrderedDict(hl[0].header)

    if verbose:
        print("@Cham: successfully load %s ..." % fp)
    return spec


def aspcapStar_read(fp, meta=False, verbose=False):
    """ read apStar fits file

    Parameters
    ----------
    fp: string
        file path
    meta: bool
        if True, attach Primary HDU header as spec.meta (OrderedDict)
    verbose: bool:
        if True, verbose.

    Returns
    -------
    spec (astropy.table.Table instance)

    Notes
    -----
    The url of the doc for aspcapStar files:
    https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/APSTAR_VERS/
    ASPCAP_VERS/RESULTS_VERS/LOCATION_ID/aspcapStar.html

    HDU0: The Primary Header
    HDU1: Spectrum array
    HDU2: Error array
    HDU3: Best fit spectrum
    HDU4: ASPCAP data table

    """
    # read apStar file
    hl = fits.open(fp)

    # construct Table instance
    spec = Table([
        Column(
            10. ** reconstruct_wcs_coord_from_fits_header(hl[1].header, 1),
            "wave"),
        Column(hl[1].data, "flux"),
        Column(hl[2].data, "flux_err"),
        Column(hl[3].data, "flux_fit")])

    # meta data
    if meta:
        spec.meta = OrderedDict(hl[0].header)

    if verbose:
        print("@Cham: successfully load %s ..." % fp)
    return spec


def test_aspcapStar_read():
    fp = "/pool/sdss/apogee_dr13/aspcapStar-r6-l30e.2-2M07332578+2044059.fits"
    spec = aspcapStar_read(fp, True)
    spec.pprint()
    print(spec.meta)
    return spec
    # spec = test_aspcapStar_read()


def test_apStar_read():
    fp = "/pool/sdss/apogee_dr13/apStar-r6-VESTA.fits"
    spec = apStar_read(fp, True)
    spec.pprint()
    print(spec.meta)
    return spec
    # spec = test_apStar_read()


def apStar_url(telescope, location_id, field, file_,
               url_header=None):
    """ apStar url generator
    which in principle is able to generate file path

    Parameters
    ----------
    telescope: string
        TELESCOPE, {"apo1m', 'apo25m'}
    location_id: int
        for 'apo1m', it's 1
        for 'apo25m', it's like PLATE
    field: string
        for 'apo1m', it's 'hip'|'calibration'|...
        for 'apo25m', it's non-sense
    file_: string
        FILE
    url_header: string
        if None|'sas', it's set to be
        "https://data.sdss.org/sas/dr13/apogee/spectro/redux/%s/stars"%version

    Returns
    -------
    url: string
        the url of apStar file

    Note
    ----
    version: string
        currently it's 'r6' @20161031
    """

    if url_header is None or url_header is "sas":
        url_header = ("https://data.sdss.org/sas/dr13/apogee"
                      "/spectro/redux/r6/stars")

    url_header = url_header.strip()
    telescope = telescope.strip()
    field = field.strip()
    file_ = file_.strip()

    if telescope == "apo1m":
        # apo1m
        url = "%s/%s/%s/%s" % (url_header, telescope, field, file_)
    elif telescope == "apo25m":
        # apo25m
        url = "%s/%s/%s/%s" % (url_header, telescope, location_id, file_)
    else:
        raise(ValueError("@Cham: This is not an option!"))
    return url


def aspcapStar_url(location_id, file_, url_header=None):

    """ aspcapStar url generator
    which in principle is able to generate file path

    Parameters
    ----------
    location_id: int
        for 'apo1m', it's 1
        for 'apo25m', it's like PLATE
    file_: string
        FILE
    url_header: string
        if None|'sas', it's set to be
        "https://data.sdss.org/sas/dr13/apogee/spectro/redux/%s/stars/l30e/l30e.2"%version

    Returns
    -------
    url: string
    the url of apStar file

    Note
    ----
    version: string
        currently it's 'r6' @20161031
    """

    if url_header is None or url_header is "sas":
        url_header = ("https://data.sdss.org/sas/dr13/apogee"
                      "/spectro/redux/r6/stars/l30e/l30e.2")

    url_header = url_header.strip()
    file_ = file_.strip()

    try:
        url = "%s/%s/%s" % (url_header, location_id, file_)
    except:
        raise (ValueError("@Cham: This is not an option!"))

    return url


def apStar_download(url, file_path, verbose=False,
                    username="sdss", password="2.5-meters"):
    """ apStar file downloading utils
    which in principle is able to download everything from a valid url

    Parameters
    ----------
    url: string
        the url of the target
    file_path: string
        the path of the file to be saved
    verbose:
        if True, print status

    Returns
    -------
    status: bool
        True if success.

    """

    # # if exists, do nothing
    # if os.path.exists(file_path):
    #     try:
    #         fits.open(file_path)
    #         return True
    #     except Exception:
    #         pass

    # credentials for sdss
    p = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    p.add_password(None, url, username, password)
    handler = urllib.request.HTTPBasicAuthHandler(p)
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    # request data
    status = True
    try:
        local_file_path, header = urllib.request.urlretrieve(url, file_path)
    except Exception:
        status = False

    # verbose
    if verbose:
        if status:
            print("@Cham: success: {0}".format(url))
        else:
            print("@Cham: failed: {0}".format(url))

    return status


def mkdir_loop(file_path, n_loop=3, verbose=True):
    """ a weak version of os.makedirs()
    which may avoid infinite loop

    Parameters
    ----------
    file_path: string
        file path
    n_loop: int
        make n-th parent directory possible to be created
    verbose: bool
        if True, verbose

    Returns
    -------
    bool

    """
    dirname = os.path.dirname(file_path)
    if n_loop > 0:

        if os.path.exists(dirname):
            # if dirname exists
            if verbose:
                print("@Cham: [n_loop=%s] dir exists: %s " % (n_loop, dirname))
            return True
        elif os.path.exists(os.path.dirname(dirname)):
            # if dirname doesn't exist, but dirname(dirname) exists --> mkdir
            if verbose:
                print("@Cham: [n_loop=%s] mkdir: %s ..." % (n_loop, dirname))
            os.mkdir(dirname)
            return True
        else:
            # if dirname(dirname) doesn't exists
            if mkdir_loop(dirname, n_loop-1):
                return mkdir_loop(file_path, n_loop)
            else:
                return False
    else:
        if verbose:
            print("@Cham: [n_loop=%s] unable to mkdir %s ..." % (n_loop, dirname))
        return False
