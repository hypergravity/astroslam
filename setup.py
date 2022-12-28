import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='astroslam',
    version="1.2022.1228.1",
    author='Bo Zhang',
    author_email='bozhang@nao.cas.cn',
    description=('A forward model using SVR to estimate stellar parameters'
                 ' from spectra.'),  # short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/hypergravity/astroslam',
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=["Development Status :: 5 - Production/Stable",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 "Programming Language :: Python :: 3.9",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Topic :: Scientific/Engineering :: Astronomy"],
    package_dir={'slam': 'slam',
                 'extern': 'slam/extern'},
    package_data={'slam': ['data/*.csv']},
    # include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'astropy',
        'laspec',
        'scikit-learn==1.2.0',
        'joblib==1.2.0',
        'pandas',
        'emcee',
        'lmfit',
        'ipyparallel'
    ]
)
