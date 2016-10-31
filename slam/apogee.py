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
import urllib2
import numpy as np
from astropy.table import Table
from joblib import load, dump, Parallel, delayed


__all__ = ['apStar_url', 'apStar_download', 'mkdir_loop']


def apStar_url(telescope, location_id, field, file_,
               version='r6', url_header=None):
    """ apStar url generator
    which in principle is able to generate file path

    Parameters
    ----------
    telescope: string
        TELESCOPE, {'apo1m', 'apo25m'}
    location_id: int
        for 'apo1m', it's 1
        for 'apo25m', it's like PLATE
    field: string
        for 'apo1m', it's 'hip'|'calibration'|...
        for 'apo25m', it's non-sense
    file_: string
        FILE
    version: string
        currently it's 'r6' @20161031
    url_header: string
        if None|'sas', it's set to be
        "https://data.sdss.org/sas/dr13/apogee/spectro/redux/%s/stars"%version

    Returns
    -------
    url: string
        the url of apStar file

    """

    if url_header is None or url_header is 'sas':
        url_header = ("https://data.sdss.org/sas/"
                      "dr13/apogee/spectro/redux/%s/stars") % version

    url_header = url_header.strip()
    telescope = telescope.strip()
    field = field.strip()
    file_ = file_.strip()

    if telescope == 'apo1m':
        # apo1m
        url = "%s/%s/%s/%s" % (url_header, telescope, field, file_)
    elif telescope == 'apo25m':
        # apo25m
        url = "%s/%s/%s/%s" % (url_header, telescope, location_id, file_)
    else:
        raise(ValueError("@Cham: This is not an option!"))
    return url


def apStar_download(url, file_path, verbose=True):
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
    status = True
    try:
        response = urllib2.urlopen(url)
        f = open(file_path, 'w+')
        f.write(response.read())
        f.close()
    except:
        status = False
    if verbose:
        if status:
            print("@Cham: success: %s" % url)
        else:
            print("@Cham: failed for url ""%s""" % url)
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
