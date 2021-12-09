#!/usr/bin/env python

from distutils.core import setup, Extension
import os
import sysconfig

# compiler flags
extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args = ["-fopenmp"] + ["-std=c++11"] + extra_compile_args

module1 = Extension('owls',
        include_dirs = ['/tools/include','/tools/lib/python2.7/site-packages/numpy/core/include'],
        library_dirs = ['/tools/lib'],
        libraries = ['gsl','gslcblas'],
        sources = ['owls.cpp'],
        extra_compile_args = ['-fopenmp']+['-std=c++11'],
        extra_link_args = ['-lgomp'],
	language="c++"
        )

setup(name = 'owls',
        version = '1.0',
        description = "owls features",
        ext_modules = [module1])
