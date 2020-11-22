import ctypes
from scipy.linalg.cython_blas cimport (
    ssyrk as tsyrk,
    sgemm as tgemm,
    sgemv as tgemv,
    ssymv as tsymv,
    ssymm as tsymm,
    ssyr as tsyr,
    scopy as tcopy,
    sdot as tdot,
    saxpy as taxpy,
    sscal as tscal,
    )
from scipy.linalg.cython_lapack cimport (
    sposv as tposv,
    spotri as tpotri,
    spotrf as tpotrf,
    ssyev as tsyev
    )
from libc.math cimport fabs as fabs_t
### TODO: change to fabsf once new cython version is released

ctypedef float real_t
C_real_t = ctypes.c_float


include "linreg_untyped.pxi"
