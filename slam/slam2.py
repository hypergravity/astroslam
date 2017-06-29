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
- Slam2 class

"""

import os
from tempfile import NamedTemporaryFile
import time
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from joblib import load, dump, Parallel, delayed
from sklearn.model_selection import train_test_split

from .diagnostic import compare_labels, single_pixel_diagnostic
from .hyperparameter import (summarize_hyperparameters_to_table,
                             summarize_table,
                             hyperparameter_grid_stats)
from .mcmc import predict_label_mcmc
from .predict import predict_labels, predict_labels_chi2, predict_spectrum
from .standardization import standardize, standardize_ivar
from .train2 import train_multi_pixels, train_single_pixel
from .utils import convolve_mask, uniform, getsize, sizeof
from .parallel import launch_ipcluster_dv, reset_dv, print_time_cost

__all__ = ['Slam2']


class Slam2(object):
    """ This defines Slam class """
    # training data
    wave = np.zeros((0, 0))
    tr_flux = np.zeros((0, 0))
    tr_ivar = np.zeros((0, 0))
    tr_labels = np.zeros((0, 0))
    tr_mask = np.zeros((0, 0), bool)
    ivar_eps = 1.

    # training data scalers
    tr_flux_scaler = np.zeros((0, 0))
    tr_ivar_scaler = np.zeros((0, 0))
    tr_labels_scaler = np.zeros((0, 0))

    # dimentions of data
    n_obs = 0
    n_pix = 0
    n_dim = 0

    ccoef = None
    init_kwargs = {}
    heal_kwargs = {}

    # SVR result list
    model = "untrained"
    sms = []
    hyperparams = Table(data=[np.zeros(0)] * 3,
                        names=['C', 'gamma', 'epsilon'])
    scores = np.zeros((0,))
    nmse = None
    stats_train = None
    stats_test = None

    trained = False
    sample_weight = None
    ind_all_bad = None

    uniform_dict = None
    uniform_picked = None

    _total_size = 0

    # ####################### #
    #     Basic functions     #
    # ####################### #

    def __init__(self, wave, tr_flux, tr_ivar, tr_labels, scale=True,
                 robust=False, mask_conv=(1, 2), flux_bounds=(1e-3, 1e2),
                 ivar_eps=1e0):
        """ initialize the Slam instance with tr_flux, tr_ivar, tr_labels

        Parameters
        ----------
        wave: 1D ndarray
            spectral wavelength
        tr_flux: ndarray with a shape of (n_obs x n_pix)
            training flux (RV-corrected, normalized)
        tr_ivar: ndarray with a shape of (n_obs x n_pix)
            training ivar
        tr_labels: ndarray with a shape of (n_obs x n_dim)
            training labels
        scale: bool, default True
            if True, scale the input flux and labels
        robust: bool, default False
            if True, use mean and std to scale flux and labels,
            otherwise, use (p84+p16)/2 and the (p84-p16)/2
            Only use this when there are extreme values, 
            or it will affect the scores!
            

        Returns
        -------
        Slam instance

        """

        # input data assertions
        try:
            # assert input data are 2-d np.ndarray
            assert isinstance(wave, np.ndarray) and wave.ndim == 1
            assert isinstance(tr_flux, np.ndarray) and tr_flux.ndim == 2
            assert isinstance(tr_ivar, np.ndarray) and tr_ivar.ndim == 2
            assert isinstance(tr_labels, np.ndarray) and tr_labels.ndim == 2

            # assert input data shape consistency
            assert tr_flux.shape == tr_ivar.shape
            assert tr_flux.shape[0] == tr_labels.shape[0]

            # to make dataset valid
            tr_flux, tr_ivar, tr_labels = self.heal_the_world(
                tr_flux, tr_ivar, tr_labels, mask_conv=mask_conv,
                flux_bounds=flux_bounds, ivar_eps=ivar_eps)
            self.init_kwargs = dict(scale=scale,
                                    robust=robust,
                                    mask_conv=mask_conv,
                                    flux_bounds=flux_bounds,
                                    ivar_eps=ivar_eps)
            self.heal_kwargs = dict(mask_conv=mask_conv,
                                    flux_bounds=flux_bounds,
                                    ivar_eps=ivar_eps)
            self.ivar_eps = ivar_eps

        except:
            raise (ValueError(
                "@SLAM: input data error, go back and check your data!"))

        # if scale: do standardization
        if scale:
            # assign attributes
            self.wave = wave
            self.tr_flux = tr_flux
            self.tr_ivar = tr_ivar
            self.tr_labels = tr_labels
            self.tr_mask = tr_ivar > 0

            # standardization -->
            # a robust way to set the scale_ to be 0.5*(16, 84) percentile
            self.tr_flux_scaler, self.tr_flux_scaled = \
                standardize(tr_flux, tr_ivar, robust=robust)
            self.tr_ivar_scaler, self.tr_ivar_scaled = \
                standardize_ivar(tr_ivar, self.tr_flux_scaler)
            self.tr_labels_scaler, self.tr_labels_scaled = \
                standardize(tr_labels, None, robust=robust)
            # TODO: here the weight of tr_labels should be passed in

        # if not scale, assume the input data is already scaled
        else:
            # assign attributes
            self.wave = wave
            self.tr_flux = tr_flux
            self.tr_ivar = tr_ivar
            self.tr_labels = tr_labels
            self.tr_mask = tr_ivar > 0

            # without standardization
            self.tr_flux_scaled = tr_flux
            self.tr_ivar_scaled = tr_ivar
            self.tr_labels_scaled = tr_labels

        # update dimensions
        self.__update_dims__()

        return

    @property
    def total_size(self):
        self._total_size = getsize(self)
        return self._total_size

    # this is slow but gets you the size of every attribute
        def size(self, unit='mb', verbose=False,
                 key_removed=["size", "size_total_mb"]):
            return sizeof(self, unit=unit, verbose=verbose,
                          key_removed=key_removed)

    @staticmethod
    def init_from_keenan(k):
        """ initiate a Slam instance from TheKeenan instance
        To guarantee it works, you need to install TheKeenan package,
        especially in the case that Keenan instance is load from dump file.

        Parameters
        ----------
        k: string
            the path of the dump file

        Returns
        -------
        a Slam instance

        Examples
        --------
        >>> from joblib import load
        >>> from slam.slam import Slam
        >>> dump_path = './keenan.dump'
        >>> k = load(dump_path)
        >>> s = Slam.init_from_keenan(k)

        """

        # initiate Slam
        s = Slam2(k.wave, k.tr_flux, k.tr_ivar, k.tr_labels)

        # get the __dict__ attribute
        k_keys = k.__dict__.keys()

        # delete used keys
        keys_to_be_del = ["wave", "tr_flux", "tr_ivar", "tr_labels"]
        for key in keys_to_be_del:
            k_keys.remove(key)

        # copy other keys
        for key in k_keys:
            s.__setattr__(key, k.__getattribute__(key))

        # return Slam instance
        return s

    # ####################### #
    #     update info         #
    # ####################### #

    def __update_dims__(self, verbose=True):
        """ update data dimensions """
        # record old data dimensions
        n_obs_old, n_pix_old, n_dim_old = self.n_obs, self.n_pix, self.n_dim
        # assign new data dimensions
        self.n_obs, self.n_pix = self.tr_flux_scaled.shape
        self.n_dim = self.tr_labels_scaled.shape[1]
        # verbose
        if verbose:
            print("")
            print("@SLAM: updating data dimensions!")
            print("----------------------------------")
            print("n_obs: %s --> %s" % (n_obs_old, self.n_obs))
            print("n_pix: %s --> %s" % (n_pix_old, self.n_pix))
            print("n_dim: %s --> %s" % (n_dim_old, self.n_dim))
            print("----------------------------------")

    def __update_hyperparams__(self):
        """ update hyper-parameters """
        self.hyperparams = summarize_hyperparameters_to_table(self.sms)
        summarize_table(self.hyperparams)
        return

    # ####################### #
    #     print info          #
    # ####################### #

    def __repr__(self):
        repr_strs = [
            "======================================",
            "Slam2 instance:",
            "======================================",
            "tr_flux............: ( %s x %s )" % self.tr_flux.shape,
            "tr_ivar............: ( %s x %s )" % self.tr_ivar.shape,
            "tr_labels..........: ( %s x %s )" % self.tr_labels.shape,
            "--------------------------------------",
            "tr_flux_scaled.....: ( %s x %s )" % self.tr_flux_scaled.shape,
            "tr_ivar_scaled.....: ( %s x %s )" % self.tr_ivar_scaled.shape,
            "tr_labels_scaled...: ( %s x %s )" % self.tr_labels_scaled.shape,
            "--------------------------------------",
            "scale..............: {}".format(self.init_kwargs["scale"]),
            "robust.............: {}".format(self.init_kwargs["robust"]),
            "mask_conv..........: {}".format(self.init_kwargs["mask_conv"]),
            "flux_bounds........: {}".format(self.init_kwargs["flux_bounds"]),
            "ivar_eps...........: {}".format(self.init_kwargs["ivar_eps"]),
            "--------------------------------------",
            "model..............: {}".format(self.model),
            "sms................: list[%s]" % len(self.sms),
            "scores.............: list[%s]" % len(self.scores),
            "hyper-parameters...: Table[length=%s]" % len(self.hyperparams),
            "trained............: %s" % self.trained,
            "======================================",
        ]
        return "\n".join(repr_strs)

    def hyperparams_summary(self, keys=None):
        """ summarize the hyper-parameter table

        Parameters
        ----------
        mask: None | bool array
            if not None, only unmasked rows are summarized

        """

        if keys is None:
            return None
        elif len(keys) == 0:
            return None
        else:
            t = Table(None, names=keys)
            for i in range(self.n_pix):
                row_ = []
                for k in keys:
                    row_.append(self.sms[i].regressor.__getattribute__(k))
                t.add_row()
            self.hyperparams = t

        return

    def hyperparameter_grid_stats(self, pivot=("param_C", "param_gamma"),
                                  n_jobs=10, verbose=10):
        self.stats_train, self.stats_test = hyperparameter_grid_stats(
            self.sms, pivot=pivot, n_jobs=n_jobs, verbose=verbose)
        return self.stats_train, self.stats_test

    def pprint(self, mask=None):
        """ print info about self & hyper-parameters """

        print(self.__repr__())
        self.hyperparams_summary(mask=mask)

        return

    # ####################### #
    #     IO utils            #
    # ####################### #

    def save_dump(self, filepath, overwrite=False, *args, **kwargs):
        """ save Slam object to dump file using joblib

        Parameters
        ----------
        filepath: string
            file path
        overwrite: bool
            If True, overwrite the file directly.

        *args, **kwargs:
            extra parameters are passed to joblib.dump()

        """
        # check file existence
        if os.path.exists(filepath) and not overwrite:
            raise (IOError("@Slam: file exists! [%s]" % filepath))
        else:
            # the joblib.dump() will overwrite file in default
            dump(self, filepath, *args, **kwargs)
            return

    @classmethod
    def load_dump(cls, filepath):
        """ load Slam instance from dump file

        Parameters
        ----------
        filepath: string
            the dump file path

        Returns
        -------
        Slam instance / arbitrary python object

        Example
        -------
        >>> k = Slam.load_dump("./slam.dump")

        """
        # check file existence
        try:
            assert os.path.exists(filepath)
        except:
            raise (IOError("@Slam: file does not exist! [%s]" % filepath))

        return load(filepath)

    def save_dump_sms(self, filepath, overwrite=False, *args, **kwargs):
        """ [NOT RECOMMENDED] save only (wave, svrs) to dump file

        Parameters
        ----------
        filepath: string
            file path
        overwrite: bool
            If True, overwrite the file directly.

        *args, **kwargs:
            extra parameters are passed to joblib.dump()

        Example
        -------
        >>> s.save_dump_svrs("./slam_sms.dump")

        """
        # check file existence
        if os.path.exists(filepath) and not overwrite:
            raise (IOError("@SLAM: file exists! [%s]" % filepath))
        else:
            # the joblib.dump() will overwrite file in default
            dump((self.wave, self.sms), filepath, *args, **kwargs)
            return

    @classmethod
    def load_dump_sms(cls, filepath):
        """ [NOT RECOMMENDED] initialize Slam instance with only *svrs* data

        Parameters
        ----------
        filepath: string
            the dump file path

        Returns
        -------
        A Slam instance
        flux, ivar and labels will be automatically filled with np.zeros

        Example
        -------
        >>> s = Slam.load_dump_sms("./slam_svrs.dump")
        >>> print(s)

        """
        wave, sms = load(filepath)
        n_pix = len(wave)
        s = Slam2(wave,
                  np.zeros((10, n_pix)),
                  np.zeros((10, n_pix)),
                  np.zeros((10, n_pix)),
                  scale=False)
        s.sms = sms
        s.trained = True
        return s

    # ####################### #
    #     training            #
    # ####################### #
    # TODO: train_single_pixel
    def train_single_pixel(self, i_train, sample_weight=None, cv=10, **kwargs):
        """ train single pixel

        Parameters
        ----------
        i_train: int
            the pixel that will be trained
        sample_weight: ndarray
            weight of each pixel
        cv: int
            cv-fold cross-validation

        Returns
        -------
        svr, score

        """
        svr, score = train_single_pixel(
            self.tr_labels_scaled,
            self.tr_flux_scaled[:, i_train].reshape(-1, 1),
            sample_weight=sample_weight.reshape(-1, 1),
            cv=cv,
            **kwargs)

        return svr, score

    def train_pixels(self, profile=None, targets="all", temp_dir=None,
                     sample_weight_scheme="bool",
                     model="nn", method="simple", param_grid=None, cv=8,
                     scoring="neg_mean_squared_error",
                     n_jobs=10, verbose=10, **kwargs):
        """ train pixels usig SVR

        Parameters
        ----------
        profile: str | None
            if str, use that ipcluster profile
            if None, simply parallel in SMP
        targets: list | slice | int | "all" 
            the targets of remote cluster
        temp_dir: str
            the directory to save temp files
            make sure it's shared between controllers and engines
        sample_weight_scheme: string
            sample weight scheme for training {"alleven", "bool", "ivar"}
        cv: int
            if cv>1, cv-fold Cross-Validation will be performed
        method: {"simple" | "grid" | "rand"}
            simple: directly use user-defined hyper-parameters
            grid: grid search for optimized hyper-parameters
            rand: randomized search for optimized hyper-parameters
        n_jobs: int
            number of jobs that will be launched simultaneously
        verbose:
            verbose level
        **kwargs:
            will be passed to the svr.fit() method

        Returns
        -------
        self.svrs: list
            a list of SVR results
        self.trained: bool
            will be set True

        """
        # determine sample_weight
        assert sample_weight_scheme in ("alleven", "bool", "ivar")

        if sample_weight_scheme is "alleven":
            # all even (some bad pixels do disturb!)
            sample_weight = np.ones_like(self.tr_flux_scaled)

        elif sample_weight_scheme is "bool":
            # 0|1 scheme for training flux (recommended)
            sample_weight = self.tr_mask.astype(np.float)
            ind_all_bad = np.sum(sample_weight, axis=0) < 1.
            for i_pix in np.arange(sample_weight.shape[1]):
                if ind_all_bad[i_pix]:
                    # this pixel is all bad
                    # reset sample weight to 1
                    sample_weight[:, i_pix] = 1.
            self.ind_all_bad = ind_all_bad

        elif sample_weight_scheme is "ivar":
            # according to ivar (may cause bias due to sampling)
            sample_weight = self.tr_ivar_scaled

        self.sample_weight = sample_weight

        # training
        if profile is None:
            # paprallel in SMP
            results = train_multi_pixels(
                self.tr_labels_scaled,
                [y for y in self.tr_flux_scaled.T],
                [sw_ for sw_ in self.sample_weight.T],
                model=model,
                method=method,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                **kwargs)

            # clear & store new results
            self.sms = []
            self.scores = []
            for svr, score in results:
                self.sms.append(svr)
                self.scores.append(score)
            self.scores = np.array(self.scores)

        else:
            time1 = time.time()

            # parallel in ipcluster
            dv = launch_ipcluster_dv(profile=profile, targets=targets,
                                     block=True)
            kwargs["model"] = model
            kwargs["method"] = method
            kwargs["param_grid"] = param_grid
            kwargs["scoring"] = scoring
            kwargs["cv"] = cv
            kwargs["method"] = method
            kwargs["n_jobs"] = 1
            kwargs["verbose"] = verbose

            # assert temp_dir exists
            if temp_dir is None:
                print("@Slam: make sure engines and controllers share"
                      " [/tmp]!)")
            else:
                try:
                    assert os.path.exists(temp_dir)
                except AssertionError as ae:
                    print("@Slam: [{}] doen't exist!".format(temp_dir))
                    raise ae

            # dump training data
            with NamedTemporaryFile(dir=temp_dir) as f:
                dump_source = f.name
            print("@Slam: dumping training data to {} ...".format(dump_source))
            dump((self.tr_labels_scaled, self.tr_flux_scaled,
                  self.sample_weight, kwargs), dump_source)

            # load training data in engines
            dv.execute("import os")
            dv.execute("import tempfile")
            dv.execute("from joblib import dump, load")
            dv.execute("from slam.train2 import train_multi_pixels")

            dv.scatter("ind", np.arange(self.n_pix))
            dv.push({"dump_source": dump_source, "temp_dir": temp_dir})
            print("@Slam: loading training data in engines ...")
            dv.execute("(tr_labels_scaled, tr_flux_scaled, sample_weight,"
                       " kwargs) = load(dump_source)")
            print("@Slam: training in engines ...")
            print("@Slam: take a cup of coffee and have a break ...")
            print("@Slam: DO NOT INTERRUPT! Or you have to RESTART ipcluster!")
            dv.execute("results = train_multi_pixels(tr_labels_scaled,"
                       "[y for y in tr_flux_scaled[:, ind].T],"
                       "[sw_ for sw_ in sample_weight[:, ind].T],"
                       "**kwargs)")

            # dump training results in engines
            dv.execute("f = tempfile.NamedTemporaryFile(dir=temp_dir)")
            dv.execute("dump_path = f.name")
            dv.execute("f.close()")
            dv.execute("dump((results, ind), dump_path)")

            # initiate svrs and scores
            svrs = [None for _ in range(self.n_pix)]
            scores = np.zeros((self.n_pix,), float)

            # gather results
            dump_paths = dv.gather("dump_path")
            print("@Slam: gathering results ...")
            for dump_path in dump_paths:
                results, ind = load(dump_path)
                for i in range(len(ind)):
                    svrs[ind[i]], scores[ind[i]] = results[i]

            # remove dump files
            print("@Slam: removing dump source {} ...".format(dump_source))
            os.remove(dump_source)
            print("@Slam: removing dump files ...")
            dv.execute("os.remove(dump_path)")

            # clear variables in engines
            print("@Slam: clear variables in engines ...")
            dv.execute("del(results, tr_labels_scaled, tr_flux_scaled, "
                       "sample_weight)")

            # store new results
            self.sms = svrs
            self.scores = scores

            # finishing
            time2 = time.time()
            print("@Slam: it took so long [{}] ..."
                  "".format(print_time_cost(time2 - time1)))
            print("@Slam: training finished!")

        # update hyper-parameters
        self.__update_hyperparams__()

        # set trained to True
        self.trained = True
        self.nmse = None
        return

    # ####################### #
    #     predicting          #
    # ####################### #

    # TODO: prediction functions forms should be confirmed

    def predict_labels(self, X0, test_flux, test_ivar=None, mask=None,
                       flux_scaler=True, labels_scaler=True, **kwargs):
        """ predict labels for a given test spectrum (single)

        Parameters
        ----------
        X0 : ndarray (1 x n_dim)
            the initial guess of predicted label
        test_flux : ndarray (n_pix, )
            test flux array
        test_ivar : ndarray (n_pix, )
            test ivar array
        mask : bool ndarray (n_pix, )
            manual mask, False pixels are not evaluated for speed up
        flux_scaler : scaler object
            flux scaler. if False, it doesn't perform scaling
        labels_scaler : scaler object
            labels scaler. if False, it doesn't perform scaling
        kwargs :
            extra parameters passed to *minimize()*
            **tol** should be specified by user according to n_pix

        """

        # if scale, set scalers
        if flux_scaler:
            flux_scaler = self.tr_flux_scaler
        else:
            flux_scaler = None

        if labels_scaler:
            labels_scaler = self.tr_labels_scaler
        else:
            labels_scaler = None

        # mask default
        if mask is None:
            mask = np.ones_like(test_flux, dtype=np.bool)

        # test_ivar default
        if test_ivar is None:
            test_ivar = np.ones_like(test_flux, dtype=np.float)

        # test_ivar normalization
        test_ivar /= np.nansum(test_ivar[mask])

        # predict labels
        X_pred = predict_labels(
            X0, self.sms, test_flux, test_ivar=test_ivar, mask=mask,
            flux_scaler=flux_scaler, labels_scaler=labels_scaler, **kwargs)

        return X_pred

    def predict_labels_quick(self, test_flux, test_ivar,
                             tplt_flux=None, tplt_ivar=None,
                             tplt_labels=None, n_sparse=1,
                             profile=None, targets="all", temp_dir="/tmp",
                             n_jobs=1, verbose=False):
        """ a quick chi2 search for labels """

        test_flux, test_ivar = self.heal_the_world(
            test_flux, test_ivar, **self.heal_kwargs)

        if profile is None:
            if tplt_flux is None and tplt_labels is None and tplt_ivar is None:
                # use default tplt_flux & tplt_labels
                X_quick = predict_labels_chi2(self.tr_flux[::n_sparse, :],
                                              self.tr_ivar[::n_sparse, :],
                                              self.tr_labels[::n_sparse, :],
                                              test_flux, test_ivar,
                                              n_jobs=n_jobs, verbose=verbose)
            else:
                # use user-defined tplt_flux & tplt_labels
                X_quick = predict_labels_chi2(tplt_flux[::n_sparse, :],
                                              tplt_ivar[::n_sparse, :],
                                              tplt_labels[::n_sparse, :],
                                              test_flux, test_ivar,
                                              n_jobs=n_jobs, verbose=verbose)
        else:
            # timing
            time1 = time.time()

            # initiate ipcluster
            dv = launch_ipcluster_dv(profile=profile, targets=targets,
                                     block=True)

            # dump data
            with NamedTemporaryFile(dir=temp_dir) as f:
                dump_path = f.name
            if tplt_flux is None and tplt_labels is None and tplt_ivar is None:
                print("@Slam: dumping data to {} ... ".format(dump_path))
                dump((self.tr_flux[::n_sparse, :], self.tr_ivar[::n_sparse, :],
                      self.tr_labels[::n_sparse, :], test_flux, test_ivar),
                     dump_path)
            else:
                print("@Slam: dumping data to {} ... ".format(dump_path))
                dump((tplt_flux[::n_sparse, :], tplt_ivar[::n_sparse, :],
                      tplt_labels[::n_sparse, :], test_flux, test_ivar,),
                     dump_path)

            # push data
            dv.push({"dump_path": dump_path})
            dv.execute("from slam.predict import predict_labels_chi2")
            dv.execute("from joblib import load, dump")
            print("@Slam: loading data in engines ...")
            dv.execute("tplt_flux, tplt_ivar, tplt_labels, test_flux, "
                       "test_ivar = load(dump_path)")
            dv.scatter("ind", np.arange(test_flux.shape[0], dtype=int))

            # predict labels
            print("@Slam: engines are working ...")
            print("@Slam: take a cup of coffee and have a break ...")
            print("@Slam: DO NOT INTERRUPT! Or you have to RESTART ipcluster!")
            dv.execute("X_quick = predict_labels_chi2(tplt_flux, tplt_ivar, "
                       "tplt_labels, test_flux[ind], test_ivar[ind], "
                       "n_jobs=1, verbose=10)")

            # gather data
            print("@Slam: gathering data ...")
            X_quick = dv.gather("X_quick")

            # remove dump file
            print("@Slam: removing dump file {} ... ".format(dump_path))
            os.remove(dump_path)

            # finishing
            time2 = time.time()
            print("@Slam: it took so long [{}] ..."
                  "".format(print_time_cost(time2 - time1)))

        return X_quick

    def predict_labels_ipc(self, X_init, test_flux, test_ivar, mask=None,
                           profile="default", targets="all", temp_dir=None,
                           count_size=True, reset=True, *args, **kwargs):
        """ parallel using ipcluster, make sure there is an ipcluster available
        
        Parameters
        ----------
        X_init:
            initial guess of parameters
        test_flux:
            flux array
        test_ivar:
            ivar array
        profile:
            ipcluster profile
        targets:
            the targets of remote cluster
        temp_dir:
            temp file directory
        reset
            if True, reset engines
        args, kwargs
            will be passed to Slam.predict_labels_multi()

        Returns
        -------

        """
        test_flux, test_ivar = self.heal_the_world(
            test_flux, test_ivar, **self.heal_kwargs)

        # timing
        time1 = time.time()

        # force n_jobs = 1 in engines
        kwargs["n_jobs"] = 1

        # initiate ipcluster
        dv = launch_ipcluster_dv(profile=profile, targets=targets, block=True,
                                 max_engines=test_flux.shape[0])

        # import modules in ipcluster
        dv.execute("import os")
        dv.execute("import tempfile")
        dv.execute("from joblib import dump, load")
        dv.execute("from slam.train import train_multi_pixels")

        # save Slam instance
        with NamedTemporaryFile(dir=temp_dir) as f:
            fp = f.name
        print("@Slam: saving Slam instance to {} ...".format(fp))
        self.save_dump(fp, overwrite=True)

        # estimate size of Slam
        if count_size:
            print("@Slam: size of me is about {} MB ".format(
                os.path.getsize(fp) / 1024 ** 2))

        # load Slam instance
        print("@Slam: loading Slam instance from {} ...".format(fp))
        dv.execute("from joblib import load")
        dv.execute("s = load('{}')".format(fp))

        # scatter data
        print("@Slam: scattering & pushing data to engines ...")
        dv.scatter("X_init", X_init)
        dv.scatter("test_flux", test_flux)
        dv.scatter("test_ivar", test_ivar)
        if mask is None:
            dv.push({"mask": mask})
        elif mask.shape == test_flux.shape:
            dv.scatter("mask", mask)
        else:
            dv.push({"mask": mask})
        dv.push({"args": args, "kwargs": kwargs})
        print("@Slam: engines are working ...")
        dv.execute("X_pred = s.predict_labels_multi("
                   "X_init, test_flux, test_ivar, *args, **kwargs)")
        print("@Slam: take a cup of coffee and have a break ...")
        print("@Slam: DO NOT INTERRUPT! Or you have to RESTART ipcluster!")
        print("@Slam: gathering results ...")
        X_pred = dv.gather("X_pred")

        time2 = time.time()
        print("@Slam: it took so long [{}] ..."
              "".format(print_time_cost(time2 - time1)))

        if reset:
            reset_dv(dv)

        return X_pred

    # in this method, do not use scaler defined in predict_labels()
    def predict_labels_multi(self, X0, test_flux, test_ivar=None, mask=None,
                             model_ivar=None, flux_eps=None, ivar_eps=1e0,
                             flux_scaler=True, ivar_scaler=True,
                             labels_scaler=True, n_jobs=1, verbose=False,
                             **kwargs):
        """ predict labels for a given test spectrum (multiple)

        Parameters
        ----------
        X0 : ndarray (1 x n_dim)
            the initial guess of predicted label
        test_flux : ndarray (n_test, n_pix)
            test flux array
        test_ivar : ndarray (n_test, n_pix)
            test ivar array
        mask : bool ndarray (n_test, n_pix)
            manual mask, False pixels are not evaluated for speed up
        model_ivar: ndarray (n_pix,), default None
            the model error in scaled space.
            if None, 0 would be adopted.
            usually 1/(-NMSE) could be used.
        flux_scaler : scaler object
            flux scaler. if False, it doesn't perform scaling
        ivar_scaler : scaler object
            ivar scaler. if False, it doesn't perform scaling
        labels_scaler : scaler object
            labels scaler. if False, it doesn't perform scaling
        n_jobs: int
            number of processes launched by joblib
        verbose: int
            verbose level

        kwargs :
            extra parameters passed to *minimize()*
            **tol** should be specified by user according to n_pix

        NOTE
        ----
        ** all input should be 2D array or sequential **

        """
        if "profile" in kwargs.keys():
            if kwargs["profile"] is None:
                kwargs.pop("profile")
            else:
                raise AssertionError(
                    "@Slam: use Slam.predict_labels_ipc instead!")

        test_flux, test_ivar = self.heal_the_world(
            test_flux, test_ivar, **self.heal_kwargs)

        # 0. determine n_test
        n_test = test_flux.shape[0]

        # 1. default scalers
        if flux_scaler:
            flux_scaler = self.tr_flux_scaler
        else:
            flux_scaler = None

        if ivar_scaler:
            ivar_scaler = self.tr_ivar_scaler
        else:
            ivar_scaler = None

        if labels_scaler:
            labels_scaler = self.tr_labels_scaler
        else:
            labels_scaler = None

        # 2. scale test_flux
        if flux_scaler is not None:
            test_flux = flux_scaler.transform(test_flux)

        # 3. set default mask, test_ivar
        # mask must be set here!
        if mask is None:
            # no mask is set
            mask = np.ones_like(test_flux, dtype=np.bool)
        elif mask.ndim == 1 and len(mask) == test_flux.shape[1]:
            # if only one mask is specified
            mask = np.array([mask for _ in range(n_test)])
        mask &= test_ivar > 0

        # 4. scale test_ivar
        # test_ivar must be set here!
        if test_ivar is None:
            # test_ivar=None, directly set test_ivar
            test_ivar = np.ones_like(test_flux, dtype=np.float)
        # test_ivar is not None
        elif ivar_scaler is not None:
            # do scaling for test_ivar
            test_ivar = ivar_scaler.transform(test_ivar)
            # else:
            # don't do scaling for test_ivar

        # 5. update test_ivar : negative ivar set to 0
        # test_ivar = np.where((test_ivar >= 0.) * (np.isfinite(test_ivar)),
        #                      test_ivar, np.zeros_like(test_ivar))
        test_ivar = np.where(
            np.logical_and(test_ivar >= ivar_eps, np.isfinite(test_ivar)),
            test_ivar, np.zeros_like(test_ivar))

        # 5'. take into account model error
        if model_ivar is not None:
            # fix model_ivar
            model_ivar = np.where(
                np.logical_and(model_ivar > ivar_eps, np.isfinite(model_ivar)),
                model_ivar, 0.)
            # calculate total_ivar
            test_ivar = np.where(
                np.logical_or(model_ivar < ivar_eps, test_ivar < ivar_eps), 0.,
                test_ivar * model_ivar / (test_ivar + model_ivar))
        # fix total_ivar
        test_ivar = np.where(np.isfinite(test_ivar), test_ivar, 0.)

        # 6. update mask for low ivar pixels
        # test_ivar_threshold = np.array(
        #     [np.median(_[_ > 0]) * 0.05 for _ in test_ivar]).reshape(-1, 1)
        # mask = np.where(test_ivar < test_ivar_threshold,
        #                 np.zeros_like(mask, dtype=np.bool), mask)
        #
        # This is NON-PHYSICAL !
        # Since the test_ivar is SCALED, could not cut 0.05 median!
        if flux_eps is not None:
            mask = np.logical_and(mask, test_flux > flux_eps)
        else:
            mask = np.logical_and(mask, test_ivar > 0.)
        print("@SLAM: The spectra with fewest pixels unmasked is "
              "[{}/{}]".format(np.min(np.sum(mask, axis=1)), self.n_pix))

        # 7. test_ivar normalization
        # test_ivar /= np.sum(test_ivar, axis=1).reshape(-1, 1)
        test_ivar /= np.percentile(test_ivar, 90, axis=1).reshape(-1, 1)

        assert test_flux.shape == test_ivar.shape
        assert test_flux.shape == mask.shape

        # 8. if you want different initial values ...
        if X0.ndim == 1:
            # only one initial guess is set
            X0 = X0.reshape(1, -1).repeat(n_test, axis=0)
        elif X0.shape[0] == 1:
            # only one initial guess is set, but 2D shape
            X0 = X0.reshape(1, -1).repeat(n_test, axis=0)

        if labels_scaler is not None:
            X0 = labels_scaler.transform(X0)

        # 9. loop predictions
        X_pred = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(predict_labels)(
                X0[i].reshape(1, -1), self.sms, test_flux[i],
                test_ivar=test_ivar[i], mask=mask[i],
                flux_scaler=None, ivar_scaler=None, labels_scaler=None,
                **kwargs) for i in range(n_test)
        )
        X_pred = np.array(X_pred)

        # 10. scale X_pred back if necessary
        if labels_scaler is not None:
            X_pred = labels_scaler.inverse_transform(X_pred)

        return X_pred

    # in this method, do not use scaler defined in predict_labels()
    def predict_labels_mcmc(self, X0, test_flux, test_ivar=None,
                            model_ivar=None, mask=None,
                            flux_eps=None, ivar_eps=1e0,
                            flux_scaler=True, ivar_scaler=True,
                            labels_scaler=True, n_jobs=1, verbose=False,
                            X_lb=None, X_ub=None,
                            n_walkers=10, n_burnin=200, n_run=500, threads=1,
                            return_chain=False, mcmc_run_max_iter=3, mcc=0.4,
                            prompts=None,
                            profile="default", targets="all", temp_dir=None,
                            **kwargs):
        """ predict labels for a given test spectrum (multiple)

        Parameters
        ----------
        X0 : ndarray (1 x n_dim)
            the initial guess of predicted label
        test_flux : ndarray (n_test, n_pix)
            test flux array
        test_ivar : ndarray (n_test, n_pix)
            test ivar array
        mask : bool ndarray (n_test, n_pix)
            manual mask, False pixels are not evaluated for speed up
        flux_scaler : scaler object
            flux scaler. if False, it doesn't perform scaling
        labels_scaler : scaler object
            labels scaler. if False, it doesn't perform scaling
        n_jobs: int
            number of processes launched by joblib
        verbose: int
            verbose level

        kwargs :
            extra parameters passed to *minimize()*
            **tol** should be specified by user according to n_pix

        NOTE
        ----
        ** all input should be 2D array or sequential **

        """

        test_flux, test_ivar = self.heal_the_world(
            test_flux, test_ivar, **self.heal_kwargs)

        # 0. determine n_test
        n_test = test_flux.shape[0]

        # 1. default scalers
        if flux_scaler:
            flux_scaler = self.tr_flux_scaler
        else:
            flux_scaler = None

        if ivar_scaler:
            ivar_scaler = self.tr_ivar_scaler
        else:
            ivar_scaler = None

        if labels_scaler:
            labels_scaler = self.tr_labels_scaler
        else:
            labels_scaler = None

        # 2. scale test_flux
        if flux_scaler is not None:
            test_flux = flux_scaler.transform(test_flux)

        # 3. set default mask, test_ivar

        # mask must be set here!
        if mask is None:
            # no mask is set
            mask = np.ones_like(test_flux, dtype=np.bool)
        elif mask.ndim == 1 and len(mask) == test_flux.shape[1]:
            # if only one mask is specified
            mask = np.array([mask for _ in range(n_test)])

        # 4. scale test_ivar
        # test_ivar must be set here!
        if test_ivar is None:
            # test_ivar=None, directly set test_ivar
            test_ivar = np.ones_like(test_flux, dtype=np.float)
        # test_ivar is not None
        elif ivar_scaler is not None:
            # do scaling for test_ivar
            test_ivar = ivar_scaler.transform(test_ivar)
            # else:
            # don't do scaling for test_ivar

        # 5. update test_ivar : negative ivar set to 0
        test_ivar = np.where(
            np.logical_and(test_ivar >= ivar_eps, np.isfinite(test_ivar)),
            test_ivar, np.zeros_like(test_ivar))

        # 5'. take into account model error
        if model_ivar is not None:
            # fix model_ivar
            model_ivar = np.where(
                np.logical_and(model_ivar > ivar_eps, np.isfinite(model_ivar)),
                model_ivar, 0.)
            # calculate total_ivar
            test_ivar = np.where(
                np.logical_or(model_ivar < ivar_eps, test_ivar < ivar_eps), 0.,
                test_ivar * model_ivar / (test_ivar + model_ivar))

        # 6. update mask for low ivar pixels
        # test_ivar_threshold = np.array(
        #     [np.median(_[_ > 0]) * 0.05 for _ in test_ivar]).reshape(-1, 1)
        # mask = np.where(test_ivar < test_ivar_threshold,
        #                 np.zeros_like(mask, dtype=np.bool), mask)
        #
        # This is NON-PHYSICAL !
        # Since the test_ivar is SCALED, could not cut 0.05 median!
        if flux_eps is not None:
            mask = np.logical_and(mask, test_flux > flux_eps)
        else:
            mask = np.logical_and(mask, test_ivar > 0.)
        print("@SLAM: The spectra with fewest pixels unmasked is "
              "[{}/{}]".format(np.min(np.sum(mask, axis=1)), self.n_pix))

        # 7. test_ivar normalization --> not necessary
        # test_ivar /= np.sum(test_ivar, axis=1).reshape(-1, 1)
        test_ivar /= np.percentile(test_ivar, 90, axis=1).reshape(-1, 1)

        assert test_flux.shape == test_ivar.shape
        assert test_flux.shape == mask.shape

        # 8. if you want different initial values ...
        if X0.ndim == 1:
            # only one initial guess is set
            X0 = X0.reshape(1, -1).repeat(n_test, axis=0)
        elif X0.shape[0] == 1:
            # only one initial guess is set, but 2D shape
            X0 = X0.reshape(1, -1).repeat(n_test, axis=0)

        if labels_scaler is not None:
            # scale
            X0 = labels_scaler.transform(X0)

            if X_lb is not None:
                theta_lb = labels_scaler.transform(X_lb)
            else:
                theta_lb = np.ones(X0[0].shape) * -10.

            if X_ub is not None:
                theta_ub = labels_scaler.transform(X_ub)
            else:
                theta_ub = np.ones(X0[0].shape) * 10.

        else:
            # don't scale
            if X_lb is not None:
                theta_lb = X_lb
            else:
                theta_lb = np.ones(X0[0].shape) * -10.

            if X_ub is not None:
                theta_ub = X_ub
            else:
                theta_ub = np.ones(X0[0].shape) * 10.

        # default prompts for MCMC jobs
        if prompts is None:
            prompts = [i for i in range(n_test)]

        # 9. loop predictions
        if profile is None:
            results_mcmc = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(predict_label_mcmc)(
                    X0[i], self.sms, test_flux[i], test_ivar[i], mask[i],
                    theta_lb=theta_lb, theta_ub=theta_ub,
                    n_walkers=n_walkers, n_burnin=n_burnin,
                    n_run=n_run, threads=threads,
                    return_chain=return_chain,
                    mcmc_run_max_iter=mcmc_run_max_iter,
                    mcc=mcc,
                    prompt=prompts[i],
                    **kwargs) for i in range(n_test))
        else:
            # timing
            time1 = time.time()

            # initiate ipcluster
            dv = launch_ipcluster_dv(profile=profile, targets=targets,
                                     block=True, max_engines=n_test)

            # collect data
            kwargs_ = dict(theta_lb=theta_lb, theta_ub=theta_ub,
                           n_walkers=n_walkers, n_burnin=n_burnin,
                           n_run=n_run, threads=1,
                           return_chain=return_chain,
                           mcmc_run_max_iter=mcmc_run_max_iter,
                           mcc=mcc, prompt=None)
            kwargs.update(kwargs_)

            # dump data
            with NamedTemporaryFile(dir=temp_dir) as f:
                fp = f.name
            print("@Slam: dumping data to {} ...".format(fp))
            dump((X0, self.sms, test_flux, test_ivar, mask, kwargs), fp)

            # load data
            print("@Slam: loading data from {} ...".format(fp))
            dv.execute("X0, svrs, test_flux, test_ivar, mask, kwargs"
                       " = load('{}')".format(fp))
            dv.scatter("ind", np.arange(n_test))

            # predict labels
            print("@Slam: engines are working ...")
            print("@Slam: take a cup of coffee and have a break ...")
            print("@Slam: DO NOT INTERRUPT! Or you have to RESTART ipcluster!")
            dv.execute("from slam.mcmc import predict_label_mcmc")
            dv.execute("results_mcmc = [predict_label_mcmc("
                       "X0[i], svrs, test_flux[i], test_ivar[i], mask[i],"
                       "**kwargs) for i in ind]")

            # dump results
            dv.execute("import tempfile")
            dv.execute("f = tempfile.NamedTemporaryFile(dir=temp_dir)")
            dv.execute("dump_path = f.name")
            dv.execute("f.close()")
            dv.execute("dump((results_mcmc, ind), dump_path)")

            # gather results
            dump_paths = dv.gather("dump_path")
            print("@Slam: gathering results ...")
            results_mcmc = [None for i in range(n_test)]
            for dump_path in dump_paths:
                results_, ind_ = load(dump_path)
                for i in range(len(ind_)):
                    results_mcmc[ind_[i]] = results_[i]

            # remove dump files
            print("@Slam: removing dump source {} ...".format(fp))
            os.remove(fp)
            print("@Slam: removing dump files ...")
            dv.execute("os.remove(dump_path)")

            # finishing
            time2 = time.time()
            print("@Slam: it took so long [{}] ..."
                  "".format(print_time_cost(time2 - time1)))

            # reset engines
            reset_dv(dv)

        # 10. scale X_pred back if necessary
        print("@Cham: wait a minute, I'm converting results ...")

        # inverse-transform theta
        if labels_scaler is not None:
            for i in range(len(results_mcmc)):
                results_mcmc[i]["theta"] = \
                    labels_scaler.inverse_transform(results_mcmc[i]["theta"])

        # extract L M U from theta
        X_predl = np.array([r["theta"][0] for r in results_mcmc])
        X_predm = np.array([r["theta"][1] for r in results_mcmc])
        X_predu = np.array([r["theta"][2] for r in results_mcmc])

        # if flatchain is returned, inverse-transform flatchain
        if return_chain and labels_scaler is not None:
            for i in range(len(results_mcmc)):
                results_mcmc[i]["flatchain"] = \
                    labels_scaler.inverse_transform(
                        results_mcmc[i]["flatchain"])

        return X_predl, X_predm, X_predu, results_mcmc

    def predict_spectra(self, X_pred, labels_scaler=True, flux_scaler=True,
                        n_jobs=1, verbose=False):
        """ predict spectra using trained SVRs

        Parameters
        ----------
        X_pred: ndarray (n_test, n_dim)
            labels of predicted spectra

        Returns
        -------
        pred_flux: ndarray (n_test, n_pix)
            predicted spectra

        """
        # convert 1d label to 2d label
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(1, -1)

        #
        if labels_scaler:
            X_pred = self.tr_labels_scaler.transform(X_pred)

        n_pred = X_pred.shape[0]
        flux_pred = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(predict_spectrum)(self.sms, X_pred[i])
            for i in range(n_pred)
        )
        flux_pred = np.array([_[0] for _ in flux_pred])

        if flux_scaler:
            self.tr_flux_scaler.inverse_transform(flux_pred)

        return flux_pred

    # ####################### #
    #     daignostics         #
    # ####################### #

    @staticmethod
    def compare_labels(*args, **kwargs):
        return compare_labels(*args, **kwargs)

    def single_pixel_diagnostic(self,
                                i_pixel,
                                test_labels,
                                diag_dim=(0, 1),
                                labels_scaler="default",
                                flux_scaler="default"):
        if labels_scaler is "default":
            labels_scaler = self.tr_labels_scaler
        if flux_scaler is "default":
            flux_scaler = self.tr_flux_scaler

        return single_pixel_diagnostic(self.sms,
                                       i_pixel,
                                       test_labels,
                                       diag_dim=diag_dim,
                                       labels_scaler=labels_scaler,
                                       flux_scaler=flux_scaler)

    # ####################### #
    #     utils               #
    # ####################### #

    def create_mask(self, mask_init, set_range, set_val):
        """ set pixels in wave_ranges to value

        Parameters
        ----------
        mask_init:
            initial mask values
        set_range:
            2D list of ranges to be set
        set_val:
            the target values of mask, True means good, False means bad

        Returns
        -------
        modified mask

        Examples
        --------
        >>> test_mask = s.creat_mask(True, [(0, 3900)], False)

        """
        # vectorize mask_init
        if np.isscalar(mask_init):
            mask_init = np.array([mask_init for _ in self.wave])

        # loop set values
        for set_range_ in set_range:
            mask_init = np.where(np.logical_and(self.wave >= set_range_[0],
                                                self.wave <= set_range_[1]),
                                 set_val, mask_init)

        return mask_init

    # ####################### #
    #     check model         #
    # ####################### #

    def check_model_pixel(self, X_pred=None, ind_pix=None, scaler=True):
        """ check one pixel model: prediction vs training

        Parameters
        ----------
        ind_pix: ndarray (1D)
            index of pixels that will be compared
            indix are numbers, not True/False

        """
        # default X_pred
        if X_pred is None:
            X_pred = self.tr_labels

        # select pixels
        if ind_pix is None:
            # compare all pixels
            ind_pix = np.arange(self.tr_flux.shape[0])  # n_obs
        elif ind_pix.ndim > 1:
            # ind_pix is ndarray > 1D
            ind_pix = ind_pix.flatten()

        # scaler labels if necessary
        if scaler:
            X_pred = self.tr_labels_scaler.transform(X_pred)

        # predict pixels
        flux_tr = self.tr_flux[:, ind_pix]
        svrs = [self.sms[i] for i in ind_pix]
        flux_pred = predict_spectrum(svrs, X_pred)

        return flux_tr, flux_pred

    def training_nmse(self, n_jobs=1, verbose=10):
        """ return Negarive Mean Squared Error (NMSE) """
        if self.nmse is None:
            print("@SLAM: NMSE is not available and will be calculated now!")
            # if value not ready, calculate them
            ind_good_pixels = ((self.tr_ivar > 0.) *
                               (self.tr_flux > 0.) *
                               np.isfinite(self.tr_ivar) *
                               np.isfinite(self.tr_flux))
            sample_weight = ind_good_pixels.astype(np.float)
            r = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(nmse)(
                self.sms[i], self.tr_labels_scaled, self.tr_flux_scaled[:, i],
                sample_weight[:, i]) for i in range(self.n_pix))
            self.nmse = np.array(r, float)

        return self.nmse

    def automask(self, flux=None, ivar=None, min_num_pix=0.6, ivar_eps=1e0):
        if isinstance(min_num_pix, int):
            # absolute value
            pass
        elif isinstance(min_num_pix, float):
            # relative value
            min_num_pix = np.round(self.n_obs * min_num_pix)
        if flux is None:
            flux = self.tr_flux
        if ivar is None:
            ivar = self.tr_ivar

        return (flux > 0) & (ivar > ivar_eps) & \
               (np.sum(self.tr_ivar > ivar_eps, axis=0) > min_num_pix)

    # ############## #
    # plotting tools #
    # ############## #
    @property
    def correlation_coefs(self):

        if self.ccoef is None:
            ccoef = np.zeros_like(self.wave, float)

            for i in range(self.n_pix):
                ind_good = self.tr_mask[:, i]
                ccoef[i] = np.abs(
                    np.corrcoef(self.tr_flux[ind_good, i],
                                self.tr_labels[ind_good, 1])[0, 1])
            self.ccoef = ccoef

        return self.ccoef

    def plot_correlation_coefs(self, fontsize=20):

        plt.rcParams.update({"font.size": fontsize})

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plt.plot(self.wave, self.correlation_coefs)

        return fig, ax

    def plot_training_performance(self, fontsize=20, mse_hline=None):
        plt.rcParams.update({"font.size":fontsize})
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.sca(axs[0])
        plt.plot(self.wave, self.tr_flux_scaler.mean_, label="flux mean")
        plt.plot(self.wave, self.tr_flux_scaler.scale_, label="flux scale")
        plt.legend(loc=5, framealpha=.3)

        fig.sca(axs[1])
        plt.plot(self.wave, -self.scores, label="cross validated MSE")
        plt.plot(self.wave, -self.training_nmse(30), label="training MSE")
        if mse_hline is not None:
            plt.hlines(mse_hline, self.wave[0], self.wave[-1], linestyle="--",
                       color="gray")
        plt.legend(loc=5, framealpha=.3)

        fig.sca(axs[2])
        plt.plot(self.wave, np.log10(self.hyperparams["C"]),
                 label="$\\log(C)$")
        plt.plot(self.wave, np.log10(self.hyperparams["gamma"]),
                 label="$\\log(\gamma)$")
        plt.plot(self.wave, np.log10(self.hyperparams["epsilon"]),
                 label="$\\log(\epsilon)$")
        plt.legend(loc=5, framealpha=.3)

        plt.xlabel("$\lambda$ ($\\rm \AA$)")
        plt.xlim(self.wave[[0, -1]])

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)

        return fig, axs

    @staticmethod
    def heal_the_world(flux, ivar, labels=None, mask_conv=(1, 2),
                       flux_bounds=(1e-3, 1e2), ivar_eps=1e0):
        """ heal the flux, ivar and label array
        
        Parameters
        ----------
        flux: 2d numpy.ndarray
            flux array        
        ivar: 2d numpy.ndarray
            ivar array
        labels: 2d numpy.ndarray
            label array
        mask_conv:
            if not None, convolve mask
        flux_bounds:
            bounds of flux
        ivar_eps:
            the lowest ivar value allowed

        Returns
        -------

        """
        # assert input data are 2-d np.ndarray
        assert isinstance(flux, np.ndarray) and flux.ndim == 2
        assert isinstance(ivar, np.ndarray) and ivar.ndim == 2
        if labels is not None:
            assert isinstance(labels, np.ndarray) and labels.ndim == 2

        # assert input data shape consistency
        assert flux.shape == ivar.shape
        if labels is not None:
            assert flux.shape[0] == labels.shape[0]

        # to make dataset valid
        if flux_bounds is not None:
            ind_flux_in_bounds = (flux > flux_bounds[0]) & (flux < flux_bounds[1])
            ind_flux_finite = np.isfinite(flux) & np.isfinite(ivar)
            ind_ivar_valid = ivar > ivar_eps
            ind_ivar_reset = np.logical_not(ivar == 0.) & np.logical_not(
                ind_flux_in_bounds & ind_flux_finite & ind_ivar_valid)
            ind_flux_reset = np.logical_not(ind_flux_finite)
            # not in bounds, or infinite, or ivar too small
            sum_flux_reset = np.sum(ind_flux_reset, axis=1)
            sum_ivar_reset = np.sum(ind_ivar_reset, axis=1)
            print("=====================================================")
            print("@Slam.heal_the_world: IVAR of {} spectra need to be reset"
                  "".format(np.sum(sum_ivar_reset > 0)))
            print("=====================================================")
            for i in np.where(sum_ivar_reset > 0)[0]:
                print("({}, {}), ".format(i, sum_ivar_reset[i]), end="")
            print("")
            print("=====================================================")
            print("")
            print("=====================================================")
            print("@Slam.heal_the_world: FLUX of {} spectra need to be reset"
                  "".format(np.sum(sum_flux_reset > 0)))
            print("=====================================================")
            for i in np.where(sum_flux_reset > 0)[0]:
                print("({}, {}), ".format(i, sum_flux_reset[i]), end="")
            print("")
            print("=====================================================")

            flux = np.where(ind_flux_reset, 0., flux)
            ivar = np.where(ind_ivar_reset, 0., ivar)

        if mask_conv is not None:
            for i in range(ivar.shape[0]):
                ivar[i] *= convolve_mask(
                    ivar[i] > 0, kernel_size_coef=.25,
                    kernel_size_limit=mask_conv, sink_region=None)

        # if labels are bad
        if labels is not None:
            ind_bad_labels = np.any(np.logical_not(np.isfinite(labels)),
                                    axis=1)
            n_bad_labels = np.sum(ind_bad_labels)
            if n_bad_labels > 0:
                print("=====================================================")
                print("@Slam.heal_the_world: {} labels are bad".format(
                    n_bad_labels))
                print("=====================================================")
                print(np.where(ind_bad_labels))

        if labels is None:
            return flux, ivar
        else:
            return flux, ivar, labels

    # ############## #
    # CV tools
    # ############## #

    def sub_slam(self, ind_sub):
        """  return a Slam instance initiated with a subset of training data
        
        Parameters
        ----------
        ind_sub: 1d array
            index array of subset

        Returns
        -------

        """
        return Slam2(self.wave, self.tr_flux[ind_sub], self.tr_ivar[ind_sub],
                     self.tr_labels[ind_sub], **self.init_kwargs)

    def train_test_split(self, test_size=0.1, train_size=0.9, random_state=0):
        """ separate a test sample from a Slam instance 
        
        Parameters
        ----------
        test_size: float / int
            relative / absolute number of test sample size
        train_size: float / int
            relative / absolute number of training sample size
        Returns
        -------
        
        Examples
        --------
        >>> ss, test = s.train_test_split(0.1)

        """
        tr_flux, test_flux, tr_ivar, test_ivar, tr_label, test_labels = \
            self._train_test_split(self.tr_flux, self.tr_ivar, self.tr_labels,
                                   test_size=test_size, train_size=train_size,
                                   random_state=random_state)
        return Slam2(self.wave, tr_flux, tr_ivar, tr_label,
                     **self.init_kwargs), (test_flux, test_ivar, test_labels)

    @staticmethod
    def _train_test_split(*arrays, **options):
        """ Split arrays or matrices into random train and test subsets
        
        Quick utility that wraps input validation and
        ``next(ShuffleSplit().split(X, y))`` and application to input data
        into a single call for splitting (and optionally subsampling) data in a
        oneliner.
        Read more in the :ref:`User Guide <cross_validation>`.
        Parameters
        ----------
        *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse
            matrices or pandas dataframes.
        test_size : float, int, or None (default is None)
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the test split. If
            int, represents the absolute number of test samples. If None,
            the value is automatically set to the complement of the train size.
            If train size is also None, test size is set to 0.25.
        train_size : float, int, or None (default is None)
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.
        random_state : int or RandomState
            Pseudo-random number generator state used for random sampling.
        stratify : array-like or None (default is None)
            If not None, data is split in a stratified fashion, using this as
            the class labels.
        Returns
        -------
        splitting : list, length=2 * len(arrays)
            List containing train-test split of inputs.
            .. versionadded:: 0.16
                If the input is sparse, the output will be a
                ``scipy.sparse.csr_matrix``. Else, output type is the same as the
                input type.
        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = np.arange(10).reshape((5, 2)), range(5)
        >>> X
        array([[0, 1],
               [2, 3],
               [4, 5],
               [6, 7],
               [8, 9]])
        >>> list(y)
        [0, 1, 2, 3, 4]
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.33, random_state=42)
        ...
        >>> X_train
        array([[4, 5],
               [0, 1],
               [6, 7]])
        >>> y_train
        [2, 0, 3]
        >>> X_test
        array([[2, 3],
               [8, 9]])
        >>> y_test
        [1, 4]
        """
        return train_test_split(*arrays, **options)

    def update(self):
        """ uodate to the new version """
        s = Slam2(self.wave, self.tr_flux, self.tr_ivar, self.tr_labels,
                  **self.init_kwargs)

        """ This defines Slam class """
        s.ccoef = self.ccoef

        # SVR result list
        s.sms = [_.update() for _ in self.sms]
        s.hyperparams = self.hyperparams

        s.scores = self.scores
        s.nmse = self.nmse
        s.stats_train = self.stats_train
        s.stats_test = self.stats_test

        s.trained = self.trained
        s.sample_weight = self.sample_weight
        s.ind_all_bad = self.ind_all_bad
        return s

    def uniform(self, bins, n_pick=3, ignore_out=False, digits=8):
        """ make a uniform sample --> index stored in Slam.uniform_good
        
        Parameters
        ----------
        bins: list of arrays
            bins in each dim
        n_pick: int
            how many to pick in each bin
        digits: int
            digits to form string
        ignore_out: bool
            if True, kick stars out of bins
            if False, raise error if there is any star out of bins 
            
        Examples
        --------
        >>> s.uniform([np.arange(3000, 6000, 100), np.arange(-1, 5, .2),
        >>>     np.arange(-5, 1, .1)], n_pick=1, ignore_out=False)

        Returns
        -------
        index of selected sub sample

        """
        self.uniform_dict = uniform(self.tr_labels, bins=bins, n_pick=n_pick,
                                    digits=digits, ignore_out=ignore_out)
        self.uniform_picked = self.uniform_dict["uniform_picked"]
        return self.uniform_picked


def nmse(svr, X, y, sample_weight=None):
    """ return NMSE for svr, X, y and sample_weight """
    if sample_weight is None:
        sample_weight = np.ones_like(y, int)

    ind_use = sample_weight > 0
    if np.sum(ind_use) > 0:
        X_ = X[ind_use]
        y_ = y[ind_use]
        # sample_weight_ = sample_weight[ind_use]
        return -np.mean(np.square(svr.predict(X_) - y_))
    else:
        return 0.


def _test_repr():
    wave = np.arange(5000, 6000)
    tr_flux = np.random.randn(10, 1000)
    tr_ivar = np.random.randn(10, 1000)
    tr_labels = np.random.randn(10, 3)
    k = Slam2(wave, tr_flux, tr_ivar, tr_labels)
    print(k)


if __name__ == "__main__":
    _test_repr()
