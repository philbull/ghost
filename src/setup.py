from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
    name = "ghost",
    version = "0.1",
    description = "ghost analytic galaxy-halo model",
    ext_modules = [ 
                Extension(
                          "ghost",
                          ["band_spec.c", "pyghost.c", "ghmock.c"],
                          libraries = ['m', 'gsl', 'gslcblas', 'gomp'],
                          depends = ['ghmock.h'],
                          extra_compile_args=['-O4', '-fopenmp',]
                          ) ],
    include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs(),
)
