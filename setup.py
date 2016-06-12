from setuptools import setup
from Cython.Build import cythonize

setup(name='orbital simulator',
      ext_modules=cythonize('orbital/*.pyx'))
