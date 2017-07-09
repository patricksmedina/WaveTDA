# from setuptools import setup, find_packages
# from codecs import open
# from os import path
#
# here = path.abspath(path.dirname(__file__))
#
# # Get the long description from the README file
# with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
#     long_description = f.read()
#
# setup(
#     name='wavetda',
#     version='0.1.0',
#     description='Python implementation of WaveTDA with Cython connectivity to BayesTDA.',
#     long_description=long_description,
#
#     # The project's main homepage.
#     url='https://github.com/patricksmedina/WaveTDA',
#
#     # Author details
#     author='Patrick S Medina',
#     author_email='psmedinadev@gmail.com',
#
#     # Choose your license
#     license='MIT',
#
#     # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
#     classifiers=[
#         'Development Status :: 3 - Alpha',
#         'Intended Audience :: End Users/Desktop',
#         'Intended Audience :: Science/Research',
#         'Topic :: Scientific/Engineering :: Information Analysis',
#         'Topic :: Scientific/Engineering :: Mathematics',
#         'License :: OSI Approved :: MIT License',
#         'Programming Language :: Python :: 2.7',
#         'Programming Language :: Python :: 3.6',
#         'Programming Language :: C++',
#         'Programming Language :: Cython'
#     ],
#
#     keywords='bayesian regression topological data analysis persistence kernels',
#
#
#     packages=find_packages(exclude=['contrib', 'docs', 'tests']),
#
#     # Alternatively, if you want to distribute just a my_module.py, uncomment
#     # this:
#     #   py_modules=["my_module"],
#
#     # List run-time dependencies here.  These will be installed by pip when
#     # your project is installed. For an analysis of "install_requires" vs pip's
#     # requirements files see:
#     # https://packaging.python.org/en/latest/requirements.html
#     install_requires=['peppercorn'],
#
#     # List additional groups of dependencies here (e.g. development
#     # dependencies). You can install these using the following syntax,
#     # for example:
#     # $ pip install -e .[dev,test]
#     extras_require={
#         'dev': ['check-manifest'],
#         'test': ['coverage'],
#     },
#
#     # If there are data files included in your packages that need to be
#     # installed, specify them here.  If using Python 2.6 or less, then these
#     # have to be included in MANIFEST.in as well.
#     package_data={
#         'sample': ['package_data.dat'],
#     },
#
#     # Although 'package_data' is the preferred approach, in some case you may
#     # need to place data files outside of your packages. See:
#     # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
#     # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
#     data_files=[('my_data', ['data/data_file'])],
#
#     # To provide executable scripts, use entry points in preference to the
#     # "scripts" keyword. Entry points provide cross-platform support and allow
#     # pip to create the appropriate form of executable for the target platform.
#     entry_points={
#         'console_scripts': [
#             'sample=sample:main',
#         ],
#     },
# )

# from distutils.core import setup, Extension
# from Cython.Build import cythonize
# import os
#
# here = os.path.abspath(os.path.dirname(__file__))
#
# ext = Extension("wavetda.statistics.bayes_regression",
#                 sources=[os.path.join(here,
#                                       "wavetda/_source/bayes_regression.pyx"),
#                          os.path.join(here,
#                                       "wavetda/_source/_bayesregression.cpp")
#                 ],
#                 extra_compile_args=['-std=c++11'],
#                 language="c++")
#
# setup(name="bayes_regression", ext_modules=cythonize(ext))

from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include
import os

here = os.path.abspath(os.path.dirname(__file__))

ext = Extension("wavetda.statistics.bayes_regression",
                sources=[os.path.join(here,"wavetda/_source/bayes_regression.pyx")],
                include_dirs = ['.',get_include()])

setup(name="bayes_regression", ext_modules=cythonize(ext))
