from distutils.core import setup


if __name__ == '__main__':
    setup(
        name='astroslam',
        version='1.0.0',
        author='Bo Zhang',
        author_email='bozhang@nao.cas.cn',
        description=('A forward model using SVR to estimate stellar parameters'
                     ' from spectra.'),  # short description
        license='MIT',
        # install_requires=['numpy>=1.7','scipy','matplotlib','nose'],
        url='http://github.com/hypergravity/slam',
        classifiers=[
            "Development Status :: 6 - Mature",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.5",
            "Topic :: Scientific/Engineering :: Astronomy",
            "Topic :: Scientific/Engineering :: Physics"],
        package_dir={'slam': 'slam',
                     'extern': 'slam/extern'},
        packages=['slam',
                  'slam/extern'],
        package_data={'slam': ['data/*.csv']},
        # include_package_data=True,
        requires=['numpy',
                  'scipy',
                  'matplotlib',
                  'astropy',
                  'sklearn',
                  'joblib',
                  'pandas',
                  'emcee',
                  'lmfit',
                  'ipyparallel']
    )
