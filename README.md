## SLAM [![DOI](https://zenodo.org/badge/161135292.svg)](https://zenodo.org/badge/latestdoi/161135292)

Stellar LAbel Machine (SLAM) is a forward model to estimate stellar labels (e.g., Teff, logg and chemical abundances).
It is based on Support Vector Regression (SVR) which is a non-parametric regression method.

For details of **SLAM**, see [Deriving the stellar labels of LAMOST spectra with Stellar LAbel Machine (SLAM)](https://ui.adsabs.harvard.edu/abs/2020ApJS..246....9Z/abstract).
Related projects: click [here](https://ui.adsabs.harvard.edu/abs/2020ApJS..246....9Z/citations).

## Author

Bo Zhang (bozhang@nao.cas.cn)

## Home page

- [https://github.com/hypergravity/astroslam](https://github.com/hypergravity/astroslam)
- [https://pypi.org/project/astroslam/](https://pypi.org/project/astroslam/)

## Install
- for the latest **stable** version:
  - `pip install -U astroslam`
- for the latest **github** version:
  - `pip install -U git+git://github.com/hypergravity/astroslam`
- for Zenodo version
  - [https://zenodo.org/record/3461504](https://zenodo.org/record/3461504)

## Tutorial
[updated on 2020-12-02]
- A new SLAM tutorial can be found [here](https://nbviewer.jupyter.org/github/hypergravity/spectroscopy/blob/main/stellar_parameters/demo_slam/demo_slam.ipynb)
- If you are interested in SLAM or have any related questions, do not hesitate to contact me.

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
@ARTICLE{2020ApJS..246....9Z,
       author = {{Zhang}, Bo and {Liu}, Chao and {Deng}, Li-Cai},
        title = "{Deriving the Stellar Labels of LAMOST Spectra with the Stellar LAbel Machine (SLAM)}",
      journal = {\apjs},
     keywords = {Astronomical methods, Astronomy data analysis, Bayesian statistics, Stellar abundances, Chemical abundances, Fundamental parameters of stars, Catalogs, Surveys, Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2020,
        month = jan,
       volume = {246},
       number = {1},
          eid = {9},
        pages = {9},
          doi = {10.3847/1538-4365/ab55ef},
archivePrefix = {arXiv},
       eprint = {1908.08677},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020ApJS..246....9Z},
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
