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
- model:

"""


from abc import ABCMeta, abstractmethod


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




