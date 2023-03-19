#cython: language_level=2
import ctypes
from scipy.linalg.cython_blas cimport (
    dsyrk as tsyrk,
    dgemm as tgemm,
    dgemv as tgemv,
    dsymv as tsymv,
    dsymm as tsymm,
    dsyr as tsyr,
    dcopy as tcopy,
    ddot as tdot,
    daxpy as taxpy,
    dscal as tscal,
    )
from scipy.linalg.cython_lapack cimport (
    dposv as tposv,
    dpotri as tpotri,
    dpotrf as tpotrf,
    dsyev as tsyev
    )
from libc.math cimport fabs as fabs_t

ctypedef double real_t
C_real_t = ctypes.c_double


include "linreg_untyped.pxi"
