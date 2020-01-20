"""
"""

import os

try:
    from setuptools import setup, convert_path
except ImportError:
    from distutils.core import setup
    from distutils.util import convert_path

setup(name='gpemu',
      version='1.01',
      packages=['gpemu'],
      author='Nicholas Krämer',
      description='Gaussian process emulators and related methods',
      license='LICENSE.txt',
      long_description=open('README.md').read())