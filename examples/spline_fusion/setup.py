#from distutils.core import setup
from setuptools import setup
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension
import os
import numpy
import cyres


ceres_include = r'D:\cxxlibs\ceres-solver-1.11.0\include'  #"/usr/local/include/ceres/"
ceres_cfg = r'D:\cxxlibs\ceres-solver-1.11.0\build\config'
eigen_include = r'D:\cxxlibs\3rd_party\include'

ceres_lib = r"D:\cxxlibs\3rd_party\lib" #"/usr/local/lib/"

#ceres_include = "/usr/local/include/ceres/"
#eigen_choices = ["/usr/local/include/eigen3", "/usr/include/eigen3"]
#eigen_include = [x for x in eigen_choices if os.path.exists(x)][0]

ext_modules = [
    Extension(
        "wrappers",
        ["cost_functions/wrappers.pyx",],
        language="c++",
        extra_compile_args=["-Zi", "/Od"],
        #extra_link_args=["-debug",],
        include_dirs=[ceres_include, ceres_cfg, numpy.get_include(), eigen_include],
        #libraries=['ceresd','gflags', 'libglog','pydbg' ],
        libraries=['ceres','gflags', 'libglog','pydbg' ],
        library_dirs=[ceres_lib,],
        cython_include_dirs=[cyres.get_cython_include()],
    )
]

setup(
  name = 'cost_functions',
  version='0.0.1',
  cmdclass = {'build_ext': build_ext},
  ext_package = 'cost_functions',
  ext_modules = ext_modules,
)
