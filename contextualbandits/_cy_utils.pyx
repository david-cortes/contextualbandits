import numpy as np
cimport numpy as np
import ctypes
from scipy.linalg.cython_lapack cimport dpotrf, dpotri
from cython cimport boundscheck, nonecheck, wraparound

cdef extern from "cy_cpp_helpers.cpp":
    void choice_over_rows_cpp(
        double pred[],
        long outp[],
        long nrows, long ncols,
        int nthreads,
        unsigned int random_seed
    ) nogil

    void topN_byrow_cpp(
        double scores[],
        long outp[],
        long nrow,
        long ncol,
        long n,
        int nthreads
    ) nogil

    void topN_softmax_cpp(
        double scores[],
        long outp[],
        long nrow,
        long ncol,
        long n,
        int nthreads,
        unsigned long seed
    ) nogil

def _choice_over_rows(
        pred_in,
        rs,
        int nthreads
    ):
    cdef np.ndarray[double, ndim=2] pred = \
        np.require(pred_in, dtype=ctypes.c_double,
                   requirements=["C_CONTIGUOUS", "OWNDATA"])
    cdef long m = pred.shape[0]
    cdef long n = pred.shape[1]
    cdef np.ndarray[long, ndim=1] outp = np.empty(m, dtype=ctypes.c_long)
    cdef unsigned int seed = rs.integers(low=1, high=np.iinfo(np.int32).max, dtype=np.uint32)

    cdef double *ptr_p = &pred[0,0]
    cdef long *ptr_outp = &outp[0]

    with nogil:
        choice_over_rows_cpp(
            ptr_p,
            ptr_outp,
            m, n,
            nthreads,
            seed
        )
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
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
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
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        for i in range(n):
            if y[i] > 0:
                cnt_pos[node_ix[i]] += 1
            else:
                cnt_neg[node_ix[i]] += 1

def topN_byrow(
        scores_in,
        long n,
        int nthreads
    ):
    cdef np.ndarray[double, ndim=2] scores = \
        np.require(scores_in, dtype=ctypes.c_double,
                   requirements=["C_CONTIGUOUS", "OWNDATA"])
    cdef long nrow = scores.shape[0]
    cdef long m = scores.shape[1]
    cdef np.ndarray[long, ndim=2] outp = np.empty((nrow, n), dtype=ctypes.c_long)

    cdef long *ptr_outp = &outp[0,0]
    cdef double *ptr_scores = &scores[0,0]

    with nogil:
        topN_byrow_cpp(
            ptr_scores,
            ptr_outp,
            nrow,
            m,
            n,
            nthreads
        )

    return outp

def topN_byrow_softmax(
        scores_in,
        long n,
        int nthreads,
        rng
    ):
    cdef np.ndarray[double, ndim=2] scores = \
        np.require(scores_in, dtype=ctypes.c_double,
                   requirements=["C_CONTIGUOUS", "OWNDATA"])
    cdef long nrow = scores.shape[0]
    cdef long m = scores.shape[1]
    cdef np.ndarray[long, ndim=2] outp = np.empty((nrow, n), dtype=ctypes.c_long)
    cdef unsigned long seed = rng.integers(0, np.iinfo(ctypes.c_ulong).max, dtype=ctypes.c_ulong)

    cdef long *ptr_outp = &outp[0,0]
    cdef double *ptr_scores = &scores[0,0]
    with nogil:
        topN_softmax_cpp(
            ptr_scores,
            ptr_outp,
            nrow,
            m,
            n,
            nthreads,
            seed
        )
    return outp
