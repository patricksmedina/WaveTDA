from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("bayestda.bayes_regression",
                sources=["./_source/bayes_regression.pyx","./_source/_bayesregression.cpp"],
                extra_compile_args=['-std=c++11'],
                language="c++")

setup(name="bayes_regression", ext_modules=cythonize(ext))
