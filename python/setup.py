import os
import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(("state/particle_methods/argsort.pyx",
                             "state/particle_methods/stochastic_volatility.pyx",
                             "state/importance_sampling/random_effects.pyx",
                             "state/direct/subsampling.pyx"
                             )),
                             include_dirs=[numpy.get_include()]
)

