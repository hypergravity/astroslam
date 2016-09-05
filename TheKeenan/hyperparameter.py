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


def summarize_hyperparameters_to_table(svrs):
    """ summarize hyper-parameters as a Table """
    hp_array = np.array([(svr.C, svr.gamma, svr.epsilon) for svr in svrs])
    return Table(data=hp_array, names=['C', 'gamma', 'epsilon'])


def summarize_table(hpt):
    """ summarize table data """

    # simgply use pandas.DataFrame.describe()
    hpt.to_pandas().describe()

    return
