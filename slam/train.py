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
- utils for training SVRs

"""

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize
from sklearn import svm, model_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def train_single_pixel(X, y, sample_weight=None, cv=10,
                       **kwargs):
    """ train a single pixel, simply CV

    Parameters
    ----------
    X: ndarray with shape (n_obs x n_dim)
        X in sklearn notation
    y: ndarray with shape (n_obs, ) --> 1D
        y in sklearn notation
    sample_weight: ndarray with shape (n_obs, ) --> 1D
        weight for sample data
    cv: int / None
        if cv>=3, Cross-Validation will be performed to calculate MSE

    kwargs:
        extra kwargs will be passed to svm.SVR() method
        e.g., C=1.0, gamma='auto', epsilon=0.1

    Returns
    -------
    svm.SVR() instance & score
    if CV is not performed, score = np.nan

    """
    # instantiate SVR
    svr = svm.SVR(**kwargs)

    if sample_weight is None:
        sample_weight = np.ones_like(y, float)
    ind_use = sample_weight > 0
    X_ = X[ind_use]
    y_ = y[ind_use]
    sample_weight_ = sample_weight[ind_use]

    # fit data
    svr.fit(X_, y_, sample_weight=sample_weight_)

    # Cross-Validation
    if cv is None or cv < 2:
        # no cross-validation will be performed
        score = -np.mean(np.square(svr.predict(X_) - y_))
    else:
        # cross-validation will be performed to calculate MSE
        assert isinstance(cv, int) and cv >= 2
        scores = model_selection.cross_val_score(
            svr, X_, y_, scoring='neg_mean_squared_error', cv=cv)
        score = scores.mean()

    return svr, score


def train_single_pixel_grid(X, y, sample_weight=None, cv=10,
                            param_grid=None, **kwargs):
    """ train a single pixel using GridSearchCV

    Parameters
    ----------
    X: ndarray with shape (n_obs x n_dim)
        X in sklearn notation
    y: ndarray with shape (n_obs, ) --> 1D
        y in sklearn notation
    sample_weight: ndarray with shape (n_obs, ) --> 1D
        weight for sample data
    cv: int / None
        if cv>=3, Cross-Validation will be performed to calculate MSE
    param_grid: dict
        key, value pairs of hyper-parameter grids
        >>> param_grid = dict(C=2. ** np.arange(-5., 6.),
        >>>                   epsilon=[0.01, 0.05, 0.1, 0.15],
        >>>                   gamma=['auto', 0.2, 0.25, 0.3, 0.5])

    kwargs:
        extra kwargs will be passed to svm.SVR() method
        e.g., C=1.0, gamma='auto', epsilon=0.1

    Returns
    -------
    svm.SVR() instance & best hyper-parameters & score
    if CV is not performed, score = np.nan

    """

    # default param_grid
    if param_grid is None:
        param_grid = dict(C=2. ** np.arange(-5., 6.),
                          epsilon=[0.01, 0.05, 0.1, 0.15],
                          gamma=['auto', 0.2, 0.25, 0.3, 0.5])
    # instantiate SVR
    svr = svm.SVR(**kwargs)
    # perform GridSearchCV
    grid = GridSearchCV(svr, param_grid, cv=cv,
                        fit_params={'sample_weight': sample_weight},
                        scoring='neg_mean_squared_error', n_jobs=1)
    # fit data
    grid.fit(X, y)

    # return (svr, score)
    return grid, grid.best_score_


def train_single_pixel_rand(X, y, sample_weight=None, cv=10,
                            n_iter=100, param_dist=None, **kwargs):
    """ train a single pixel using RandomizedSearchCV

    Parameters
    ----------
    X: ndarray with shape (n_obs x n_dim)
        X in sklearn notation
    y: ndarray with shape (n_obs, ) --> 1D
        y in sklearn notation
    sample_weight: ndarray with shape (n_obs, ) --> 1D
        weight for sample data
    cv: int / None
        if cv>=3, Cross-Validation will be performed to calculate MSE
    n_iter: int
        the number of sampling of the random subset of hyper-parameter space
    param_dist: dict
        key, value pairs of hyper-parameter grids
        >>> param_dist = dict(C=stats.expon(scale=3),
        >>>                   gamma=stats.expon(scale=.1))

    kwargs:
        extra kwargs will be passed to svm.SVR() method
        e.g., C=1.0, gamma='auto', epsilon=0.1

    Returns
    -------
    svm.SVR() instance & best hyper-parameters & score
    if CV is not performed, score = np.nan

    """

    # default param_grid
    if param_dist is None:
        param_dist = dict(C=stats.expon(scale=3),
                          gamma=stats.expon(scale=.1))
    # instantiate SVR
    svr = svm.SVR(**kwargs)
    # perform RandomizedSearchCV
    rand = RandomizedSearchCV(svr, param_dist, n_iter=n_iter, cv=cv,
                              fit_params={'sample_weight': sample_weight},
                              scoring='neg_mean_squared_error', n_jobs=1)
    # fit data
    rand.fit(X, y)

    # return (svr, score)
    return rand, rand.best_score_


def svr_mse(hyperparam, X, y, verbose=False):
    """ Cross-Validation MES for SVR """
    gamma, C, epsilon = 10. ** np.array(hyperparam)

    # instantiate
    svr = svm.SVR(gamma=gamma, C=C, epsilon=epsilon)

    # MSE
    scores = model_selection.cross_val_score(
        svr, X, y, scoring='neg_mean_squared_error', cv=10, verbose=False)
    score = -scores.mean()

    # verbose
    if verbose:
        print(gamma, C, epsilon, score)

    return score


def train_single_pixel_mini(X, y, sample_weight=None, cv=10, **kwargs):
    """ train a single pixel using minize

    Parameters
    ----------
    X: ndarray with shape (n_obs x n_dim)
        X in sklearn notation
    y: ndarray with shape (n_obs, ) --> 1D
        y in sklearn notation
    sample_weight: ndarray with shape (n_obs, ) --> 1D
        weight for sample data
    cv: int / None
        if cv>=3, Cross-Validation will be performed to calculate MSE

    kwargs:
        extra parameters that will be passed to svm.SVR()


    Returns
    -------
    svm.SVR() instance & best hyper-parameters & score
    if CV is not performed, score = np.nan

    """
    # find optimized hyper-parameters
    hp0 = (-2., .7, -.15)
    hp = minimize(svr_mse, hp0, args=(X, y, sample_weight))
    gamma, C, epsilon = 10. ** np.array(hp)

    # specify hyper-parameters directly
    return train_single_pixel(X, y, sample_weight=sample_weight, cv=cv,
                              gamma=gamma, C=C, epsilon=epsilon, **kwargs)


def train_multi_pixels(X, ys, sample_weights, cv=1,
                       method='simple', n_jobs=1, verbose=10, **kwargs):
    """ train multi pixels

    Parameters
    ----------
    X: ndarray with shape (n_obs x n_dim)
        X in sklearn notation
    ys: ndarray with shape (n_obs x n_pix) -->
        y in sklearn notation
    sample_weights: ndarray
        weight of sample data
    cv: int
        number of fold in Cross-Validation
    method: string
        {'simple', 'grid', 'rand'}
    n_jobs: int
        number of processes that will be launched by joblib
    verbose: int
        the same as joblib.Parallel() parameter verbose
    kwargs:
        extra kwargs will be passed to svm.SVR() method

    Returns
    -------
    svm.SVR() instance

    """
    # determine method
    train_funcs = {'simple': train_single_pixel,
                   'grid': train_single_pixel_grid,
                   'rand': train_single_pixel_rand}
    train_func = train_funcs[method]

    # parallel run for SVR
    data = []
    for y, sample_weight in zip(ys, sample_weights):
        this_X = np.asarray(X, float, order='C')
        this_y = np.asarray(y, float, order='C')
        this_sw = np.asarray(sample_weight, float, order='C')
        this_ind = this_sw > 0
        data.append((this_X[this_ind], this_y[this_ind], this_sw[this_ind]))

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(train_func)(*this_data, cv=cv, **kwargs) for this_data in data)

    # return results
    return results
