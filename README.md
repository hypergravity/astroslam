## SLAM

Stellar LAbel Machine (SLAM) is a forward model to estimate stellar labels (e.g., Teff, logg and chemical abundances).
It is based on Support Vector Regression (SVR) which is a non-parametric regression method.

For details of **SLAM**, see [Zhang et al. (2019)](https://arxiv.org/abs/1908.08677).

## Author

Bo Zhang (bozhang@nao.cas.cn)

## Home page

- [https://github.com/hypergravity/astroslam](https://github.com/hypergravity/astroslam)
- [https://pypi.org/project/astroslam/](https://pypi.org/project/astroslam/)

## Install
- for the latest **stable** version:
  - `pip install astroslam`
- for the latest **github** version:
  - `pip install git+git://github.com/hypergravity/astroslam`


## Requirements

- numpy
- scipy
- matplotlib
- astropy
- scikit-learn
- joblib
- pandas
- emcee
