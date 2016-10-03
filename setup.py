from distutils.core import setup


if __name__ == '__main__':
    setup(
        name='TheKeenan',
        version='0.3.0',
        author='Bo Zhang',
        author_email='bozhang@nao.cas.cn',
        description=('A forward model using SVR to estimate stellar parameters'
                     ' from spectra.'),  # short description
        license='MIT',
        # install_requires=['numpy>=1.7','scipy','matplotlib','nose'],
        url='http://github.com/hypergravity/TheKeenan',
        classifiers=[
            "Development Status :: 6 - Mature",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.5",
            "Topic :: Scientific/Engineering :: Astronomy",
            "Topic :: Scientific/Engineering :: Physics"],
        package_dir={'TheKeenan': 'TheKeenan',
                     'extern': 'TheKeenan/extern'},
        packages=['TheKeenan',
                  'TheKeenan/extern'],
        # package_data={'starlight_wrapper': ['data/SDSS_DR10/*',
        #                                     'data/SDSS_DR10/0603/*',
        #                                     'data/executable/*',
        #                                     'data/template_mask/*',
        #                                     'data/template_base/*',
        #                                     'data/template_base/Base.BC03/*',
        #                                     'data/template_base/Base.SED.FWHM2.5/*',
        #                                     'data/template_base/Base.SED.FWHM0.3/*']},
        # include_package_data=True,
        requires=['numpy', 'scipy', 'matplotlib', 'astropy',
                  'sklearn', 'joblib', 'pandas', 'emcee']
    )
