## SLAM [![DOI](https://zenodo.org/badge/161135292.svg)](https://zenodo.org/badge/latestdoi/161135292)

Stellar LAbel Machine (SLAM) is a forward model to estimate stellar labels (e.g., Teff, logg and chemical abundances).
It is based on Support Vector Regression (SVR) which is a non-parametric regression method.

For details of **SLAM**, see [Deriving the stellar labels of LAMOST spectra with Stellar LAbel Machine (SLAM)](https://ui.adsabs.harvard.edu/abs/2019arXiv190808677Z/abstract).

**Related Projects**
1. [Exploring the spectral information content in the LAMOST medium-resolution survey (MRS)](https://ui.adsabs.harvard.edu/abs/2019arXiv191013154Z/abstract) 
2. [Tracing Kinematic and Chemical Properties of Sagittarius Stream by K-Giants, M-Giants, and BHB stars](https://ui.adsabs.harvard.edu/abs/2019arXiv190912558Y/abstract)

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
- for Zenodo version
  - [https://zenodo.org/record/3461504](https://zenodo.org/record/3461504)

## Requirements
- numpy
- scipy
- matplotlib
- astropy
- scikit-learn
- joblib
- pandas
- emcee

## How to cite
Paper:
```
@ARTICLE{2019arXiv190808677Z,
       author = {{Zhang}, Bo and {Liu}, Chao and {Deng}, Li-Cai},
        title = "{Deriving the stellar labels of LAMOST spectra with Stellar LAbel Machine (SLAM)}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = "2019",
        month = "Aug",
          eid = {arXiv:1908.08677},
        pages = {arXiv:1908.08677},
archivePrefix = {arXiv},
       eprint = {1908.08677},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190808677Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
Code:
```
@misc{https://doi.org/10.5281/zenodo.3461504,
    author = {Zhang, Bo},
    title = {hypergravity/astroslam: Stellar LAbel Machine},
    doi = {10.5281/zenodo.3461504},
    url = {https://zenodo.org/record/3461504},
    publisher = {Zenodo},
    year = {2019}
}
```

For other formats, please go to [https://search.datacite.org/works/10.5281/zenodo.3461504](https://search.datacite.org/works/10.5281/zenodo.3461504).
