#!/usr/bin/env bash
rm -rf build
rm -rf dist
#python setup.py build_ext --inplace
#python setup.py install
python setup.py sdist
pip install --force-reinstall dist/*.tar.gz