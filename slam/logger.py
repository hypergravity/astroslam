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
- Thu Jan 19 16:00:00 2017

Modifications
-------------
- Thu Jan 19 16:00:00 2017

Aims
----
- a customized logger for SLAM

"""

# verbose level:
# 1 debug
# 2 info
# 3 warning
# 4 error
# 5 critical


import logging

# configuration
format_slam = "[%(asctime)s] [%(module)s] %(levelname)s: %(message)s"
logging.basicConfig(format=format_slam, level=0)

# create a logger for SLAM
logger = logging.getLogger('SLAM')

# how to use it in other modules:
# from .logger import logger
# logger.info("msg")
