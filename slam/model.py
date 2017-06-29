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
- Thu Feb 16 17:00:00 2016

Modifications
-------------
- Thu Feb 16 17:00:00 2016

Aims
----
- SlamModel

"""


from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


METHOD_ALL = ["simple", "grid"]

MODEL_ALL = ["svr", "nusvr", "nn", "dt"]
MODEL_MAP = {'svr': SVR,
             'nusvr': NuSVR,
             'nn': MLPRegressor,
             # the three types above are recommended
             'dt': DecisionTreeRegressor,
             'abr': ensemble.AdaBoostRegressor,
             'br': ensemble.BaggingRegressor,
             'etr': ensemble.ExtraTreesRegressor,
             'gbr': ensemble.GradientBoostingRegressor,
             'rfr': ensemble.RandomForestRegressor
             }

SCORING_ALL = ["neg_mean_absolute_error",
               "neg_mean_squared_error",
               "neg_median_absolute_error"
               "r2"]


class SlamModel(object):

    regressor = None
    model = "svr"
    method = ""
    cv = 1
    scoring = "neg_mean_squared_error"

    # trained
    trained = False

    # GridSearchCV attributes
    cv_results_ = None

    def __init__(self, model="nn", method="grid",
                 param_grid=None, cv=8, scoring="neg_mean_squared_error",
                 **kwargs):

        try:
            assert model in MODEL_ALL
        except AssertionError as ae:
            print("@SlamModel: invalid kind!")
            raise ae

        try:
            assert method in METHOD_ALL
        except AssertionError as ae:
            print("@SlamModel: invalid method!")
            raise ae

        self.model = model
        self.method = method
        self.param_grid = param_grid
        self.cv = np.int(cv)
        self.scoring = scoring
        self.score_ = 0

        if self.method == "simple":
            self.regressor = MODEL_MAP[model](**kwargs)
        elif self.method == "grid":
            assert param_grid is not None
            assert self.cv > 2
            assert self.scoring in SCORING_ALL

            self.regressor = GridSearchCV(MODEL_MAP[model](**kwargs),
                                          cv=self.cv, param_grid=param_grid,
                                          scoring=self.scoring)

    def update(self):
        sm = SlamModel()
        sm.regressor = self.regressor
        sm.model = self.model
        sm.method = self.method
        sm.cv = self.cv
        sm.scoring = self.scoring
        sm.trained = self.trained
        sm.cv_results_ = self.cv_results_

        return sm

    def eval_score(self, X, y, sample_weight=None):
        if self.method == "grid":
            self.score_ = self.best_score_
        else:
            if self.model == "nn":
                self.score_ = self.regressor.score(X, y)
            else:
                self.score_ = self.regressor.score(X, y, sample_weight)

    def score(self, *args, **kwargs):
        return self.regressor.score(*args, **kwargs)

    def cross_val_score(self, X, y):
        """ retrun NMSE if cv<2 """
        if self.cv < 2:
            return mean_squared_error(y, self.predict(X), None)
        else:
            return cross_val_score(self.regressor, X, y, cv=self.cv,
                                   scoring=self.scoring).mean()

    def fit(self, X, y, weight=None):
        if weight is None:
            # support weight
            if self.model == "nn":
                self.regressor.fit(X, y)
            else:
                self.regressor.fit(X, y)

            if self.method == "grid":
                self.score_ = self.regressor.best_score_
            else:
                self.score_ = self.score(X, y)

        else:
            ind_weight = weight > 0
            # support weight
            if self.model == "nn":
                self.regressor.fit(X[ind_weight], y[ind_weight])
                if self.method == "grid":
                    self.score_ = self.regressor.best_score_
                else:
                    self.score_ = self.cross_val_score(
                        X[ind_weight], y[ind_weight])

            else:
                self.regressor.fit(X[ind_weight], y[ind_weight],
                                   weight[ind_weight])

                if self.method == "grid":
                    self.score_ = self.regressor.best_score_
                else:
                    self.score_ = self.cross_val_score(
                        X[ind_weight], y[ind_weight], weight[ind_weight])

        self.trained = True

        return

    def predict(self, X):
        return self.regressor.predict(X)

    @staticmethod
    def train(X, y, sample_weight=None, model="nn", method="grid",
              param_grid=None, cv=8, scoring="neg_mean_squared_error",
              **kwargs):
        """ train a single pixel using GridSearchCV

        Parameters
        ----------
        X: ndarray with shape (n_obs x n_dim)
            X in sklearn notation
        y: ndarray with shape (n_obs, ) --> 1D
            y in sklearn notation
        sample_weight: ndarray with shape (n_obs, ) --> 1D
            weight for sample data
        model:
            model type
        method:
            "simple" | "grid"
        param_grid: dict
            key, value pairs of hyper-parameter grids
            >>> param_grid = dict(C=2. ** np.arange(-5., 6.),
            >>>                   epsilon=[0.01, 0.05, 0.1, 0.15],
            >>>                   gamma=['auto', 0.2, 0.25, 0.3, 0.5])
        cv: int / None
            if cv>=3, Cross-Validation will be performed to calculate MSE
        scoring:
            the scoring scheme of cross validation
        kwargs:
            extra kwargs will be passed to svm.SVR() method
            e.g., C=1.0, gamma='auto', epsilon=0.1

        Returns
        -------
        svm.SVR() instance & best hyper-parameters & score
        if CV is not performed, score = np.nan

        """

        sm = SlamModel(model=model, method=method,
                       param_grid=param_grid, cv=cv, scoring=scoring,
                       **kwargs)

        # fit data
        sm.fit(X, y, sample_weight)

        return sm, sm.score_


class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict_single_spectrum(self):
        pass

    @abstractmethod
    def predict_multi_spectra(self):
        pass


def nmse(model, X, y, sample_weight=None):
    """ return NMSE for svr, X, y and sample_weight """
    if sample_weight is None:
        sample_weight = np.ones_like(y, int)

    ind_use = sample_weight > 0
    if np.sum(ind_use) > 0:
        X_ = X[ind_use]
        y_ = y[ind_use]
        # sample_weight_ = sample_weight[ind_use]
        return -np.mean(np.square(model.predict(X_) - y_))
    else:
        return 0.