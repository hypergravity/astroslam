# SLAM

Stellar LAbel Machine (SLAM) is a forward model to estimate stellar parameters (e.g., Teff, logg, [Fe/H] and chemical abundances).
It is based on Support Vector Regression (SVR), which in essential is a non-parametric regression method.

# Author

Bo Zhang (bozhang@nao.cas.cn)


# Installation

Currently, you have to download the package and install it using
`(sudo) python setup.py install`.

In the future, use `pip install slam`.


# Requirements

- numpy
- scipy
- matplotlib
- astropy
- sklearn
- joblib
- pandas
- emcee
