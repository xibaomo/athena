#!/usr/bin/env python

from distutils.core import setup, Extension

module1 = Extension('owls',
        include_dirs = ['/tools/include','/tools/lib/python2.7/site-packages/numpy/core/include'],
        library_dirs = ['/tools/lib'],
        libraries = ['gsl','gslcblas'],
        sources = ['owls.cpp'])

setup(name = 'owls',
        version = '1.0',
        description = "owls features",
        ext_modules = [module1])
