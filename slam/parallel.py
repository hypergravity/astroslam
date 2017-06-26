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
- Sun Jun 25 13:00:00 2017

Modifications
-------------
- Sun Jun 25 13:00:00 2017

Aims
----
- utils for computing in parallel

"""

from copy import deepcopy
import numpy as np
from ipyparallel import Client


def launch_ipcluster_dv(profile="default", targets="all", block=True, max_engines=None):
    # initiate ipcluster
    rc = Client(profile=profile)

    # print ipcluster information
    n_proc = len(rc.ids)
    if targets == "all":
        targets = rc.ids

    dv = rc.direct_view(targets=targets)

    # check number of engines
    # print(rc.ids, dv.targets, targets, max_engines)
    if max_engines is not None:
        if len(dv.targets) > max_engines:
            targets = deepcopy(dv.targets)
            np.random.shuffle(targets)
            targets = targets[:max_engines]
            targets.sort()

            dv = rc.direct_view(targets=targets)

    print("===================================================")
    print("@Slam: ipcluster[{}, n_engines={}/{}]".format(
        profile, len(dv.targets), n_proc))
    print("---------------------------------------------------")

    dv.block = block

    # import basic modules in ipcluster
    dv.execute("import os")
    dv.execute("import numpy as np")
    dv.execute("from joblib import Parallel, delayed, dump, load")

    # print host information
    dv.execute("host_names = os.uname()[1]").get()
    u_host_names, u_counts = np.unique(
        dv["host_names"], return_counts=True)
    for i in range(len(u_counts)):
        print("host: {} x {}".format(u_host_names[i], u_counts[i]))
    print("===================================================")

    return dv


def reset_dv(dv):
    dv.execute("import IPython\n"
               "ipy=IPython.get_ipython()\n"
               "ipy.run_line_magic(\"reset\", \" -f\")\n")
    return


def print_time_cost(dtime, unit_max="hour"):
    """ return string for delta_time """
    if dtime <= 60 * 1.5:
        dtime_str = "{:.3f} sec".format(dtime)
    elif dtime <= 60 * 60 * 1.5:
        dtime_str = "{:.3f} min".format(dtime / 60.)
    elif dtime <= (60 * 60 * 24 * 3):
        dtime_str = "{:.3f} hours".format(dtime / 3600.)
    else:
        # even larger
        if unit_max == "hour":
            dtime <= (60 * 60 * 24 * 3)
            dtime_str = "{:.3f} hours".format(dtime / 3600.)
        else:
            dtime_str = "{:.3f} days".format(dtime / 86400.)

    return dtime_str


