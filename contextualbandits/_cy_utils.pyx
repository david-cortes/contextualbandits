import numpy as np
cimport numpy as np
from cython.parallel cimport prange, threadid
import ctypes
from scipy.linalg.cython_lapack cimport dpotrf, dpotri

def _choice_over_rows(
        np.ndarray[double, ndim=2] pred,
        rs,
        int nthreads
    ):
    pred = np.require(pred, requirements=["C_CONTIGUOUS", "OWNDATA", "WRITEABLE"])
    pred[:] /= pred.sum(axis = 1, keepdims = True)
    cdef long m = pred.shape[0]
    cdef long n = pred.shape[1]
    cdef np.ndarray[long, ndim=1] outp = np.empty(m, dtype=ctypes.c_long)
    cdef np.ndarray[double, ndim=1] rnd = rs.random(size = m)
    cdef np.ndarray[double, ndim=1] cump = np.zeros(nthreads, dtype=np.float64)

    cdef double *ptr_p = &pred[0,0]
    cdef double *ptr_r = &rnd[0]
    cdef double *ptr_cump = &cump[0]
    cdef long* ptr_outp = &outp[0]
    cdef long row, col

    for row in prange(m, schedule="static", num_threads=nthreads, nogil=True):
        ptr_cump[threadid()] = 0
        for col in range(n):
            ptr_cump[threadid()] += ptr_p[col + row*n]
            if ptr_cump[threadid()] >= ptr_r[row]:
                ptr_outp[row] = col
                break
        if (col == (n-1)):
            ptr_outp[row] = col

    return outp

def _matrix_inv_symm(
        np.ndarray[double, ndim=2] X,
        double lambda_
    ):
    cdef double *ptr_X = &X[0,0]
    cdef int n = X.shape[0]
    cdef int *ptr_n = &n
    cdef char lo = 'L'
    cdef char *ptr_lo = &lo
    cdef int ignore
    cdef int *ptr_ignore = &ignore
    cdef long i
    with nogil:
        for i in range(n):
            ptr_X[i + i*n] += lambda_
        dpotrf(ptr_lo, ptr_n, ptr_X, ptr_n, ptr_ignore)
        dpotri(ptr_lo, ptr_n, ptr_X, ptr_n, ptr_ignore)

def _create_node_counters(
        np.ndarray[long, ndim=1] cnt_pos,
        np.ndarray[long, ndim=1] cnt_neg,
        np.ndarray[long, ndim=1] node_ix,
        np.ndarray[double, ndim=1] y
    ):
    cdef long *ptr_pos = &cnt_pos[0]
    cdef long *ptr_neg = &cnt_neg[0]
    cdef long *ptr_ix  = &node_ix[0]
    cdef double *ptr_y = &y[0]

    cdef long n = node_ix.shape[0]
    cdef long i
    with nogil:
        for i in range(n):
            if y[i] > 0:
                cnt_pos[node_ix[i]] += 1
            else:
                cnt_neg[node_ix[i]] += 1
