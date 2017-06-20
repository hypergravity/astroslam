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
- Mon Sep 05 12:00:00 2016

Modifications
-------------
- Mon Sep 05 12:00:00 2016

Aims
----
- SVR hyper-parameters

"""

import numpy as np
from astropy.table import Table
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from copy import deepcopy
import pandas as pd
from joblib import Parallel, delayed


__all__ = ['summarize_hyperparameters_to_table', 'summarize_table']


# ############################ #
# to summarize grid parameters #
# ############################ #
def hyperparameter_grid_stats(svrs, pivot=("param_C", "param_gamma"),
                              n_jobs=10, verbose=10):
    """ statistics for GridSearchCV results """
    stats_train = []
    stats_test = []
    r = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(hyperparameter_grid_stats_)(svr, pivot=pivot) for svr in svrs)
    for i in range(len(r)):
        stats_train.append(r[i][0])
        stats_test.append(r[i][1])
    return stats_train, stats_test


def hyperparameter_grid_stats_(svr, pivot=("param_C", "param_gamma")):
    """ statistics for GridSearchCV results """
    if isinstance(svr, GridSearchCV):
        # yes, that's it
        cvr = svr.cv_results_
        stats_train_ = deepcopy(cvr)
        stats_test_ = deepcopy(cvr)
        for k in cvr.keys():
            if k.find("test") > -1:
                stats_train_.pop(k)
            elif k.find("train") > -1:
                stats_test_.pop(k)

        if pivot is not None:
            return (
                pd.DataFrame(stats_train_).pivot(*pivot, "mean_train_score"),
                pd.DataFrame(stats_test_).pivot(*pivot, "mean_test_score"))
        else:
            return pd.DataFrame(stats_train_), pd.DataFrame(stats_test_)
    else:
        return pd.DataFrame(), pd.DataFrame()


# ######################## #
# summarize best estimator #
# ######################## #

def summarize_hyperparameters_to_table(svrs):
    """ summarize hyper-parameters as a Table

    Parameters
    ----------
    svrs: list of sklearn.svm.SVR objects
        a list of fitted SVR objets

    """
    hyperparams = []
    for svr in svrs:
        if isinstance(svr, SVR):
            hyperparams.append((svr.C, svr.gamma, svr.epsilon))
        elif isinstance(svr, GridSearchCV):
            hyperparams.append((svr.best_estimator_.C,
                                svr.best_estimator_.gamma,
                                svr.best_estimator_.epsilon))
    hp_array = np.array(hyperparams)
    return Table(data=hp_array, names=['C', 'gamma', 'epsilon'])


def summarize_table(hpt):
    """ summarize table data

    Parameters
    ----------
    hpt: astropy.table.Table
        a table of parameter

    """

    # simply use pandas.DataFrame.describe()
    print(hpt.to_pandas().describe())

    return
