from setuptools import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import numpy

import os, tempfile, subprocess, shutil

# see http://openmp.org/wp/openmp-compilers/
omp_test = r"""#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel
printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
}
"""

def has_openmp():
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    filename = r'test.c'
    file = open(filename,'w', 0)
    file.write(omp_test)
    with open(os.devnull, 'w') as fnull:
        result = subprocess.call(['cc', '-fopenmp', filename], stdout=fnull,
                                 stderr=fnull)

    file.close
    os.chdir(curdir)
    #clean up
    shutil.rmtree(tmpdir)

    return result == 0

ceres_include = r'D:\cxxlibs\ceres-solver-1.11.0\include'  #"/usr/local/include/ceres/"
ceres_cfg = r'D:\cxxlibs\ceres-solver-1.11.0\build\config'
glog_include = r'D:\cxxlibs\3rd_party\include'

ceres_lib = r"D:\cxxlibs\3rd_party\lib" #"/usr/local/lib/"
gflags_lib = r"D:\cxxlibs\3rd_party\lib"  #"/usr/local/lib/"
glog_lib = r"D:\cxxlibs\3rd_party\lib" #"/usr/local/lib/"
cholmod_lib = amd_lib = camd_lib = colamd_lib = r"D:\cxxlibs\suitesparse-metis-for-windows-master\build\lib\Release"  #"/usr/local/lib/"
ssparse_lib = r"D:\cxxlibs\suitesparse-metis-for-windows-master\build\lib\Release"
#cxsparse_lib = "/usr/local/lib/"

extra_compile_args = ['-O3']
extra_link_args = []

if 0 and has_openmp():
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-lgomp']

ext_modules = [
    Extension(
        "cyres",
        ["cyres/src/cyres.pyx", "cyres/src/cyres.pxd", "cyres/src/ceres.pxd"],
        language="c++",
        include_dirs=[ceres_include,ceres_cfg,glog_include, numpy.get_include()],
        libraries=['ceres', 'gflags', 'libglog', "libcholmod", "libcamd", "libamd", "libcolamd", "suitesparseconfig", "metis"],
        library_dirs=[ceres_lib, gflags_lib, glog_lib, cholmod_lib, amd_lib, camd_lib, colamd_lib, ssparse_lib],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
  name = 'cyres',
  version='0.0.1',
  cmdclass = {'build_ext': build_ext},
  ext_package = 'cyres',
  ext_modules = ext_modules,
  packages= ['cyres'],
  package_data={'cyres': ['src/*.pxd']},
  scripts=['scripts/cyresc']
)
