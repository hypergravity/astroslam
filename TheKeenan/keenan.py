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
- Keenan class

"""

from __future__ import print_function

import os

import numpy as np
from astropy.table import Table
from joblib import load, dump, Parallel, delayed

from .hyperparameter import summarize_hyperparameters_to_table, summarize_table
from .predict import predict_labels, predict_labels_chi2, predict_spectrum
from .standardization import standardize, standardize_ivar
from .train import train_multi_pixels
from .mcmc import predict_label_mcmc

__all__ = ['Keenan']


class Keenan(object):
    """ This defines Keenan class """
    # training data
    wave = np.zeros((0, 0))
    tr_flux = np.zeros((0, 0))
    tr_ivar = np.zeros((0, 0))
    tr_labels = np.zeros((0, 0))

    # training data scalers
    tr_flux_scaler = np.zeros((0, 0))
    tr_ivar_scaler = np.zeros((0, 0))
    tr_labels_scaler = np.zeros((0, 0))

    # dimentions of data
    n_obs = 0
    n_pix = 0
    n_dim = 0

    # SVR result list
    svrs = []
    hyperparams = Table(data=[np.zeros(0)] * 3,
                        names=['C', 'gamma', 'epsilon'])
    scores = np.zeros((0,))
    trained = False

    # ####################### #
    #     Basic functions     #
    # ####################### #

    def __init__(self, wave, tr_flux, tr_ivar, tr_labels, scale=True):
        """ initialize the Keenan instance with tr_flux, tr_ivar, tr_labels

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

        Returns
        -------
        Keenan instance

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

        except:
            raise (ValueError(
                "@Keenan: input data error, go back and check your data!"))

        # if scale: do standardization
        if scale:
            # assign attributes
            self.tr_flux = tr_flux
            self.tr_ivar = tr_ivar
            self.tr_labels = tr_labels

            # standardization
            self.tr_flux_scaler, self.tr_flux_scaled = \
                standardize(tr_flux)
            self.tr_ivar_scaler, self.tr_ivar_scaled = \
                standardize_ivar(tr_ivar, self.tr_flux_scaler)
            self.tr_labels_scaler, self.tr_labels_scaled = \
                standardize(tr_labels)

            # update dimensions
            self.__update_dims__()

        # if not scale, assume the input data is already scaled
        else:
            # assign attributes
            self.tr_flux_scaled = tr_flux
            self.tr_ivar_scaled = tr_ivar
            self.tr_labels_scaled = tr_labels

            # update dimensions
            self.__update_dims__()

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
            print("@Keenan: updating data dimensions!")
            print("----------------------------------")
            print("n_obs: %s --> %s" % (n_obs_old, self.n_obs))
            print("n_pix: %s --> %s" % (n_pix_old, self.n_pix))
            print("n_dim: %s --> %s" % (n_dim_old, self.n_dim))
            print("----------------------------------")

    def __update_hyperparams__(self):
        """ update hyper-parameters """
        self.hyperparams = summarize_hyperparameters_to_table(self.svrs)
        summarize_table(self.hyperparams)
        return

    # ####################### #
    #     print info          #
    # ####################### #

    def __repr__(self):
        repr_strs = [
            "Keenan instance:",
            "tr_flux............: ( %s x %s )" % self.tr_flux.shape,
            "tr_ivar............: ( %s x %s )" % self.tr_ivar.shape,
            "tr_labels..........: ( %s x %s )" % self.tr_labels.shape,

            "tr_flux_scaled.....: ( %s x %s )" % self.tr_flux_scaled.shape,
            "tr_ivar_scaled.....: ( %s x %s )" % self.tr_ivar_scaled.shape,
            "tr_labels_scaled...: ( %s x %s )" % self.tr_labels_scaled.shape,

            "svrs...............: list[%s]" % len(self.svrs),
            "scores.............: list[%s]" % len(self.scores),
            "hyper-parameters...: Table[length=%s]" % len(self.hyperparams),
            "trained............: %s" % self.trained,
        ]
        return '\n'.join(repr_strs)

    def hyperparams_summary(self, mask=None):
        """ summarize the hyper-parameter table

        Parameters
        ----------
        mask: None | bool array
            if not None, only unmasked rows are summarized

        """

        if mask is None:
            summarize_table(self.hyperparams)
        else:
            summarize_table(self.hyperparams[mask])

        return

    def pprint(self, mask=None):
        """ print info about self & hyper-parameters """

        print(self.__repr__())
        self.hyperparams_summary(mask=mask)

        return

    # ####################### #
    #     IO utils            #
    # ####################### #

    def save_dump(self, filepath, overwrite=False, *args, **kwargs):
        """ save Keenan object to dump file using joblib

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
            raise (IOError("@Keenan: file exists! [%s]" % filepath))
        else:
            # the joblib.dump() will overwrite file in default
            dump(self, filepath, *args, **kwargs)
            return

    @classmethod
    def load_dump(cls, filepath):
        """ load Keenan instance from dump file

        Parameters
        ----------
        filepath: string
            the dump file path

        Returns
        -------
        Keenan instance / arbitrary python object

        Example
        -------
        >>> k = Keenan.load_dump('./keenan.dump')

        """
        # check file existence
        try:
            assert os.path.exists(filepath)
        except:
            raise (IOError("@Keenan: file does not exist! [%s]" % filepath))

        return load(filepath)

    def save_dump_svrs(self, filepath, overwrite=False, *args, **kwargs):
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
        >>> k.save_dump_svrs('./keenan_svrs.dump')

        """
        # check file existence
        if os.path.exists(filepath) and not overwrite:
            raise (IOError("@Keenan: file exists! [%s]" % filepath))
        else:
            # the joblib.dump() will overwrite file in default
            dump((self.wave, self.svrs), filepath, *args, **kwargs)
            return

    @classmethod
    def load_dump_svrs(cls, filepath):
        """ [NOT RECOMMENDED] initialize Keenan instance with only *svrs* data

        Parameters
        ----------
        filepath: string
            the dump file path

        Returns
        -------
        A Keenan instance
        flux, ivar and labels will be automatically filled with np.zeros

        Example
        -------
        >>> k = Keenan.load_dump_svrs('./keenan_svrs.dump')
        >>> print(k)

        """
        wave, svrs = load(filepath)
        n_pix = len(wave)
        k = Keenan(wave,
                   np.zeros((10, n_pix)),
                   np.zeros((10, n_pix)),
                   np.zeros((10, n_pix)),
                   scale=False)
        k.svrs = svrs
        k.trained = True
        return k

    # ####################### #
    #     training            #
    # ####################### #

    def train_pixels(self, cv=10, n_jobs=10, method='simple', verbose=10,
                     *args, **kwargs):
        """ train pixels usig SVR

        Parameters
        ----------
        n_jobs: int
            number of jobs that will be launched simultaneously
        cv: int
            if cv>1, cv-fold Cross-Validation will be performed
        method: {'simple' | 'grid' | 'rand'}
            simple: directly use user-defined hyper-parameters
            grid: grid search for optimized hyper-parameters
            rand: randomized search for optimized hyper-parameters
        verbose:
            verbose level
        *args, **kwargs:
            will be passed to the svr.fit() method

        Returns
        -------
        self.svrs: list
            a list of SVR results
        self.trained: bool
            will be set True

        """
        # training
        results = train_multi_pixels(self.tr_labels_scaled,
                                     [y for y in self.tr_flux_scaled.T],
                                     [None for y in self.tr_flux_scaled.T],
                                     cv,
                                     method=method,
                                     n_jobs=n_jobs,
                                     verbose=verbose,
                                     **kwargs)

        # clear & store new results
        self.svrs = []
        self.scores = []
        for svr, score in results:
            self.svrs.append(svr)
            self.scores.append(score)

        # update hyper-parameters
        self.__update_hyperparams__()

        # set trained to True
        self.trained = True
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
            X0, self.svrs, test_flux, test_ivar=test_ivar, mask=mask,
            flux_scaler=flux_scaler, labels_scaler=labels_scaler, **kwargs)

        return X_pred

    def predict_labels_quick(self, test_flux, test_ivar,
                             tplt_flux=None, tplt_labels=None, n_sparse=1,
                             n_jobs=1, verbose=False):
        """ a quick chi2 search for labels """

        if tplt_flux is None and tplt_labels is None:
            # use default tplt_flux & tplt_labels
            X_quick = predict_labels_chi2(self.tr_flux[::n_sparse, :],
                                          self.tr_labels[::n_sparse, :],
                                          test_flux, test_ivar,
                                          n_jobs=n_jobs, verbose=verbose)
        else:
            # use user-defined tplt_flux & tplt_labels
            X_quick = predict_labels_chi2(tplt_flux[::n_sparse, :],
                                          tplt_labels[::n_sparse, :],
                                          test_flux, test_ivar,
                                          n_jobs=n_jobs, verbose=verbose)

        return X_quick

    # in this method, do not use scaler defined in predict_labels()
    def predict_labels_multi(self, X0, test_flux, test_ivar=None,
                             mask=None, flux_scaler=True, ivar_scaler=True,
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
        test_ivar = np.where(test_ivar < 0.,
                             np.zeros_like(test_ivar), test_ivar)
        test_ivar = np.where(np.isnan(test_ivar),
                             np.zeros_like(test_ivar), test_ivar)
        test_ivar = np.where(np.isinf(test_ivar),
                             np.zeros_like(test_ivar), test_ivar)

        # 6. update mask for low ivar pixels
        test_ivar_threshold = np.array(
            [np.median(_[_ > 0]) * 0.05 for _ in test_ivar]).reshape(-1, 1)
        mask = np.where(test_ivar < test_ivar_threshold,
                        np.zeros_like(mask, dtype=np.bool), mask)

        # 7. test_ivar normalization
        test_ivar /= np.sum(test_ivar, axis=1).reshape(-1, 1)

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
                X0[i].reshape(1, -1), self.svrs, test_flux[i],
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
                            mask=None, flux_scaler=True, ivar_scaler=True,
                            labels_scaler=True, n_jobs=1, verbose=False,
                            X_lb=None, X_ub=None,
                            n_walkers=10, n_burnin=200, n_run=500, threads=1,
                            return_chain=False, mcmc_run_max_iter=3, mcc=0.4,
                            prompts=None,
                            *args, **kwargs):
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
        test_ivar = np.where(test_ivar < 0.,
                             np.zeros_like(test_ivar), test_ivar)
        test_ivar = np.where(np.isnan(test_ivar),
                             np.zeros_like(test_ivar), test_ivar)
        test_ivar = np.where(np.isinf(test_ivar),
                             np.zeros_like(test_ivar), test_ivar)

        # 6. update mask for low ivar pixels
        test_ivar_threshold = np.array(
            [np.median(_[_ > 0]) * 0.05 for _ in test_ivar]).reshape(-1, 1)
        mask = np.where(test_ivar < test_ivar_threshold,
                        np.zeros_like(mask, dtype=np.bool), mask)

        # 7. test_ivar normalization --> not necessary
        # test_ivar /= np.sum(test_ivar, axis=1).reshape(-1, 1)

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

        if prompts is None:
            prompts = [i for i in range(n_test)]
        # 9. loop predictions
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(predict_label_mcmc)(
                X0[i], self.svrs, test_flux[i], test_ivar[i], mask[i],
                theta_lb=theta_lb, theta_ub=theta_ub,
                n_walkers=n_walkers, n_burnin=n_burnin,
                n_run=n_run, threads=threads,
                return_chain=return_chain,
                mcmc_run_max_iter=mcmc_run_max_iter,
                mcc=mcc,
                prompt=prompts[i],
                *args, **kwargs) for i in range(n_test)
        )

        # 10. scale X_pred back if necessary
        # print("X_pred: ", X_pred)
        print("@Cham: wait a minute, I'm converting results ...")
        print("RESULTS0: ", results[0])
        if return_chain:
            # return X_pred, flatchain
            X_predm = []
            X_predl = []
            X_predu = []
            flatchain = []
            for X_pred_, flatchain_ in results:
                X_predl.append(X_pred_[0])
                X_predm.append(X_pred_[1])
                X_predu.append(X_pred_[2])
                flatchain.append(flatchain_)

            if labels_scaler is not None:
                # scale back
                X_predl = labels_scaler.inverse_transform(np.array(X_predl))
                X_predm = labels_scaler.inverse_transform(np.array(X_predm))
                X_predu = labels_scaler.inverse_transform(np.array(X_predu))
                flatchain = [labels_scaler.inverse_transform(_) for _ in
                             flatchain]

            return X_predl, X_predm, X_predu, flatchain

        else:
            X_predm = []
            X_predl = []
            X_predu = []
            for X_pred_ in results:
                X_predl.append(X_pred_[0])
                X_predm.append(X_pred_[1])
                X_predu.append(X_pred_[2])

            if labels_scaler is not None:
                # scale back
                X_predl = labels_scaler.inverse_transform(np.array(X_predl))
                X_predm = labels_scaler.inverse_transform(np.array(X_predm))
                X_predu = labels_scaler.inverse_transform(np.array(X_predu))

            return X_predl, X_predm, X_predu

    def predict_spectra(self, X_pred, scaler=True, n_jobs=1, verbose=False):
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
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(1, -1)

        if scaler:
            X_pred = self.tr_labels_scaler.transform(X_pred)

        n_pred = X_pred.shape[0]
        flux_pred = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(predict_spectrum)(self.svrs, X_pred[i])
            for i in range(n_pred)
        )

        return np.array(flux_pred)

    # ####################### #
    #     daignostics         #
    # ####################### #

    def compare_labels(self):
        pass

    # ####################### #
    #     utils               #
    # ####################### #

    def set_mask(self, mask_init, set_range, set_val):
        """ set pixels in wave_ranges to value
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
        svrs = [self.svrs[i] for i in ind_pix]
        flux_pred = predict_spectrum(svrs, X_pred)

        return flux_tr, flux_pred


def _test_repr():
    wave = np.arange(5000, 6000)
    tr_flux = np.random.randn(10, 1000)
    tr_ivar = np.random.randn(10, 1000)
    tr_labels = np.random.randn(10, 3)
    k = Keenan(wave, tr_flux, tr_ivar, tr_labels)
    print(k)


if __name__ == '__main__':
    _test_repr()
