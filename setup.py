from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np
from sys import platform

class build_ext_subclass( build_ext ):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        if compiler == 'msvc': # visual studio
            for e in self.extensions:
                e.extra_compile_args += ['/O2', '/openmp']
        else:
            for e in self.extensions:
                e.extra_compile_args += ['-O3', '-march=native', '-fopenmp']
                e.extra_link_args += ['-fopenmp']

                if e.language == "c++":
                    e.extra_compile_args += ['-std=c++11']

            ### Remove this code if you have a mac with gcc or clang + openmp
            if platform[:3] == "dar":
                for e in self.extensions:
                    e.extra_compile_args = [arg for arg in e.extra_compile_args if arg != '-fopenmp']
                    e.extra_link_args    = [arg for arg in e.extra_link_args    if arg != '-fopenmp']
        build_ext.build_extensions(self)

setup(
    name = 'contextualbandits',
    packages = ['contextualbandits', 'contextualbandits.linreg'],
    install_requires=[
        'numpy>=1.17',
        'scipy',
        'pandas>=0.25.0',
        'scikit-learn>=0.22',
        'joblib>=0.13',
        'cython'
    ],
    version = '0.3.7',
    description = 'Python Implementations of Algorithms for Contextual Bandits',
    author = 'David Cortes',
    author_email = 'david.cortes.rivera@gmail.com',
    url = 'https://github.com/david-cortes/contextualbandits',
    keywords = 'contextual bandits offset tree doubly robust policy linucb thompson sampling',
    classifiers = [],
    cmdclass = {'build_ext': build_ext_subclass},
    ext_modules = [
        Extension("contextualbandits.linreg._wrapper_double",
                  sources=["contextualbandits/linreg/linreg_double.pyx"],
                  include_dirs=[np.get_include()]),
        Extension("contextualbandits.linreg._wrapper_float",
                  sources=["contextualbandits/linreg/linreg_float.pyx"],
                  include_dirs=[np.get_include()]),
        Extension("contextualbandits._cy_utils", language="c++",
                  sources=["contextualbandits/_cy_utils.pyx"],
                  include_dirs=[np.get_include()])
    ]
)
