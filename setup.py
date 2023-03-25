from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np
from sys import platform
import sys, os, subprocess, warnings, re

## Modify this to make the output of the compilation tests more verbose
silent_tests = not (("verbose" in sys.argv)
                    or ("-verbose" in sys.argv)
                    or ("--verbose" in sys.argv))

## Workaround for python<=3.9 on windows
try:
    EXIT_SUCCESS = os.EX_OK
except AttributeError:
    EXIT_SUCCESS = 0

class build_ext_subclass( build_ext ):
    def build_extensions(self):
        if self.compiler.compiler_type == 'msvc':
            for e in self.extensions:
                e.extra_compile_args += ['/O2', '/openmp']
        else:
            if not self.check_for_variable_dont_set_march() and not self.check_cflags_contain_arch() and not self.check_cxxflags_contain_arch():
                self.add_march_native()
            self.add_openmp_linkage() ### for c++ only

            for e in self.extensions:
                # e.extra_compile_args += ['-O3', '-march=native', '-fopenmp']
                # e.extra_link_args += ['-fopenmp']
                e.extra_compile_args += ['-O3']

                if e.language == "c++":
                    e.extra_compile_args += ['-std=c++11']

        build_ext.build_extensions(self)

    def check_cflags_contain_arch(self):
        if "CFLAGS" in os.environ:
            arch_list = [
                "-march", "-mcpu", "-mtune", "-msse", "-msse2", "-msse3",
                "-mssse3", "-msse4", "-msse4a", "-msse4.1", "-msse4.2",
                "-mavx", "-mavx2", "-mavx512"
            ]
            for flag in arch_list:
                if flag in os.environ["CFLAGS"]:
                    return True
        return False

    def check_cxxflags_contain_arch(self):
        if "CXXFLAGS" in os.environ:
            arch_list = [
                "-march", "-mcpu", "-mtune", "-msse", "-msse2", "-msse3",
                "-mssse3", "-msse4", "-msse4a", "-msse4.1", "-msse4.2",
                "-mavx", "-mavx2", "-mavx512"
            ]
            for flag in arch_list:
                if flag in os.environ["CXXFLAGS"]:
                    return True
        return False

    def check_for_variable_dont_set_march(self):
        return "DONT_SET_MARCH" in os.environ

    def add_march_native(self):
        is_apple = sys.platform[:3].lower() == "dar"
        args_march_native = ["-march=native", "-mcpu=native"]
        for arg_march_native in args_march_native:
            if self.test_supports_compile_arg(arg_march_native, with_c_comp=True):
                for e in self.extensions:
                    e.extra_compile_args.append(arg_march_native)
                break

    def add_openmp_linkage(self):
        arg_omp1 = "-fopenmp"
        arg_omp2 = "-fopenmp=libomp"
        args_omp3 = ["-fopenmp=libomp", "-lomp"]
        arg_omp4 = "-qopenmp"
        arg_omp5 = "-xopenmp"
        is_apple = sys.platform[:3].lower() == "dar"
        args_apple_omp = ["-Xclang", "-fopenmp", "-lomp"]
        args_apple_omp2 = ["-Xclang", "-fopenmp", "-L/usr/local/lib", "-lomp", "-I/usr/local/include"]
        has_brew_omp = False
        if is_apple:
            try:
                res_brew_pref = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True)
                if res_brew_pref.returncode == EXIT_SUCCESS:
                    brew_omp_prefix = res_brew_pref.stdout.decode().strip()
                    args_apple_omp3 = ["-Xclang", "-fopenmp", f"-L{brew_omp_prefix}/lib", "-lomp", f"-I{brew_omp_prefix}/include"]
                    has_brew_omp = True
            except Exception as e:
                pass


        if self.test_supports_compile_arg(arg_omp1, with_omp=True, with_c_comp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp1)
                e.extra_link_args.append(arg_omp1)
        elif is_apple and self.test_supports_compile_arg(args_apple_omp, with_omp=True, with_c_comp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-lomp"]
        elif is_apple and self.test_supports_compile_arg(args_apple_omp2, with_omp=True, with_c_comp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-L/usr/local/lib", "-lomp"]
                e.include_dirs += ["/usr/local/include"]
        elif is_apple and has_brew_omp and self.test_supports_compile_arg(args_apple_omp3, with_omp=True, with_c_comp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += [f"-L{brew_omp_prefix}/lib", "-lomp"]
                e.include_dirs += [f"{brew_omp_prefix}/include"]
        elif self.test_supports_compile_arg(arg_omp2, with_omp=True, with_c_comp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-fopenmp=libomp"]
                e.extra_link_args += ["-fopenmp"]
        elif self.test_supports_compile_arg(args_omp3, with_omp=True, with_c_comp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-fopenmp=libomp"]
                e.extra_link_args += ["-fopenmp", "-lomp"]
        elif self.test_supports_compile_arg(arg_omp4, with_omp=True, with_c_comp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp4)
                e.extra_link_args.append(arg_omp4)
        elif self.test_supports_compile_arg(arg_omp5, with_omp=True, with_c_comp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp5)
                e.extra_link_args.append(arg_omp5)

    def test_supports_compile_arg(self, comm, with_omp=False, with_c_comp=False):
        is_supported = False
        try:
            if not hasattr(self.compiler, "compiler_cxx"):
                return False
            if not isinstance(comm, list):
                comm = [comm]
            print("--- Checking compiler support for option '%s'" % " ".join(comm))
            fname = "contextualbandits_compiler_testing.cpp"
            with open(fname, "w") as ftest:
                ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
            try:
                if not isinstance(self.compiler.compiler_cxx, list):
                    cmd = list(self.compiler.compiler_cxx)
                else:
                    cmd = self.compiler.compiler_cxx
            except Exception:
                cmd = self.compiler.compiler_cxx
            if with_c_comp:
                if not isinstance(self.compiler.compiler, list):
                    cmd0 = list(self.compiler.compiler)
                else:
                    cmd0 = self.compiler.compiler
            if with_omp:
                with open(fname, "w") as ftest:
                    ftest.write(u"#include <omp.h>\nint main(int argc, char**argv) {return 0;}\n")
            try:
                val = subprocess.run(cmd + comm + [fname], capture_output=silent_tests).returncode
                is_supported = (val == EXIT_SUCCESS)
                if is_supported and with_c_comp:
                    val = subprocess.run(cmd0 + comm + [fname], capture_output=silent_tests).returncode
                    is_supported = (val == EXIT_SUCCESS)
            except Exception:
                is_supported = False
        except Exception:
            pass
        try:
            os.remove(fname)
        except Exception:
            pass
        return is_supported

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
    version = '0.3.20-3',
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
