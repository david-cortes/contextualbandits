import numpy as np
cimport numpy as np
import ctypes
from libc.string cimport memcpy, memset
from libc.math cimport isnan, isfinite
from cython cimport boundscheck, nonecheck, wraparound
from libc.stdio cimport printf

ctypedef enum CBLAS_ORDER:
    CblasRowMajor = 101
    CblasColMajor = 102

ctypedef CBLAS_ORDER CBLAS_LAYOUT

ctypedef enum CBLAS_TRANSPOSE:
    CblasNoTrans=111
    CblasTrans=112
    CblasConjTrans=113
    CblasConjNoTrans=114

ctypedef enum CBLAS_UPLO:
    CblasUpper=121
    CblasLower=122

ctypedef enum CBLAS_DIAG:
    CblasNonUnit=131
    CblasUnit=132

ctypedef enum CBLAS_SIDE:
    CblasLeft=141
    CblasRight=142

cdef void cblas_tsyrk(
        CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
        int N, int K, real_t alpha, real_t *A, int lda, real_t beta, real_t *C, int ldc
    ) nogil:
    cdef char uplo
    cdef char trans
    if (Order == CblasColMajor):
        if (Uplo == CblasUpper):
            uplo = 'U'
        else:
            uplo = 'L'

        if (Trans == CblasTrans):
            trans = 'T'
        elif (Trans == CblasConjTrans):
            trans = 'C'
        else:
            trans = 'N'

    else:
        if (Uplo == CblasUpper):
            uplo = 'L'
        else:
            uplo = 'U'

        if (Trans == CblasTrans):
            trans = 'N'
        elif (Trans == CblasConjTrans):
            trans = 'N'
        else:
            trans = 'T'

    tsyrk(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc)

cdef void cblas_tgemm(
        CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K,
        real_t alpha, real_t *A, int lda, real_t *B, int ldb, real_t beta, real_t *C, int ldc
    ) nogil:
    cdef char transA
    cdef char transB

    if (Order == CblasColMajor):
        if (TransA == CblasTrans):
            transA = 'T'
        elif (TransA == CblasConjTrans):
            transA = 'C'
        else:
            transA = 'N'

        if (TransB == CblasTrans):
            transB = 'T'
        elif (TransB == CblasConjTrans):
            transB = 'C'
        else:
            transB = 'N'

        tgemm(&transA, &transB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc)

    else:
        if (TransA == CblasTrans):
            transB = 'T'
        elif (TransA == CblasConjTrans):
            transB = 'C'
        else:
            transB = 'N'

        if (TransB == CblasTrans):
            transA = 'T';
        elif (TransB == CblasConjTrans):
            transA = 'C'
        else:
            transA = 'N'

        tgemm(&transA, &transB, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc)

cdef void cblas_tgemv(
        CBLAS_ORDER order,  CBLAS_TRANSPOSE TransA,  int m, int n,
        real_t alpha, real_t  *a, int lda,  real_t  *x, int incx,  real_t beta,  real_t  *y, int incy
    ) nogil:
    cdef char trans
    if (order == CblasColMajor):
        if (TransA == CblasNoTrans):
            trans = 'N';
        elif (TransA == CblasTrans):
            trans = 'T'
        else:
            trans = 'C'
        tgemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy)

    else:
        if (TransA == CblasNoTrans):
            trans = 'T'
        elif (TransA == CblasTrans):
            trans = 'N'
        else:
            trans = 'N'

        tgemv(&trans, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy)

cdef void cblas_tsymv(
        CBLAS_LAYOUT layout,
        CBLAS_UPLO Uplo, int N,
        real_t alpha, real_t  *A, int lda,
        real_t  *X, int incX, real_t beta,
        real_t  *Y, int incY
    ) nogil:
    cdef char ul
    if (layout == CblasColMajor):
        if (Uplo == CblasUpper):
            ul = 'U'
        else:
            ul = 'L'
    else:
        if (Uplo == CblasUpper):
            ul = 'L'
        else:
            ul = 'U'
    tsymv(&ul, &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY)

cdef void cblas_tsymm(
        CBLAS_LAYOUT layout, CBLAS_SIDE Side,
        CBLAS_UPLO Uplo, int M, int N,
        real_t alpha, real_t  *A, int lda,
        real_t  *B, int ldb, real_t beta,
        real_t  *C, int ldc
    ) nogil:
    cdef char sd
    cdef char ul

    if (layout == CblasColMajor):
        if (Side == CblasRight):
            sd = 'R'
        else:
            sd = 'L'

        if (Uplo == CblasUpper):
            ul = 'U'
        else:
            ul = 'L'

        tsymm(&sd, &ul, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc)

    else:
        if (Side == CblasRight):
            sd = 'L'
        else:
            sd = 'R'

        if (Uplo == CblasUpper):
            ul = 'L'
        else:
            ul = 'U'

        tsymm(&sd, &ul, &N, &M, &alpha, A, &lda, B, &ldb, &beta, C, &ldc)

cdef void cblas_tsyr(
        CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
        int N, real_t  alpha, real_t  *X,
        int incX, real_t  *A, int lda
    ) nogil:
    cdef char ul

    if (layout == CblasColMajor):
        if (Uplo == CblasLower):
            ul = 'L'
        else:
            ul = 'U'
        tsyr(&ul, &N, &alpha, X, &incX, A, &lda)

    else:
        if (Uplo == CblasLower):
            ul = 'U'
        else:
            ul = 'L'
        tsyr(&ul, &N, &alpha, X, &incX, A, &lda)


cdef void tsymv_dense_sp(
        real_t A[], int n_plusb,
        real_t data[], long indptr[], long indices[], long row,
        real_t outp[], bint add_bias
    ) nogil:
    cdef long i
    cdef long col
    cdef long rowA
    memset(outp, 0, n_plusb*sizeof(real_t))
    for rowA in range(n_plusb):
        for i in range(indptr[row], indptr[row+1]):
            col = indices[i]
            outp[rowA] += A[(col + rowA*n_plusb) if (col >= rowA) else (rowA + col*n_plusb)] * data[i]
    
    cdef int one = 1
    cdef real_t one_real_t = 1.
    if add_bias:
        taxpy(&n_plusb, &one_real_t, A + (n_plusb-1), &n_plusb, outp, &one)

cdef real_t tdot_dense_sp(
        real_t *dense_vec, long row,
        real_t data[], long indptr[], long indices[]
    ) nogil:
    cdef real_t res = 0
    cdef long i
    for i in range(indptr[row], indptr[row+1]):
        res += dense_vec[indices[i]] * data[i]
    return res

cdef real_t* get_ptr_real_t(np.ndarray[real_t, ndim=1] arr):
    return &arr[0]

cdef long* get_ptr_long(np.ndarray[long, ndim=1] arr):
    return &arr[0]

def cast_csr(Xcsr):
    if Xcsr.data.dtype != C_real_t:
        Xcsr.data = Xcsr.data.astype(C_real_t)
    if Xcsr.indptr.dtype != ctypes.c_long:
        Xcsr.indptr = Xcsr.indptr.astype(ctypes.c_long)
    if Xcsr.indices.dtype != ctypes.c_long:
        Xcsr.indices = Xcsr.indices.astype(ctypes.c_long)

def fit_model_noinv(
        np.ndarray[real_t, ndim=2] X,
        np.ndarray[real_t, ndim=1] y,
        np.ndarray[real_t, ndim=1] w,
        Xcsr = None,
        bint add_bias=1,
        real_t lam = 1.,
        bint calc_inv=0
    ):
    cdef int n = X.shape[1] if Xcsr is None else Xcsr.shape[1]
    cdef int n_plusb = n + <int>add_bias
    cdef int m = X.shape[0] if Xcsr is None else Xcsr.shape[0]
    cdef np.ndarray[real_t, ndim=2] XtX = np.empty((n_plusb,n_plusb), dtype=C_real_t)
    cdef np.ndarray[real_t, ndim=1] XtY = np.empty(n_plusb, dtype=C_real_t)

    update_matrices_noinv(
        X, y, w,
        XtX, XtY,
        Xcsr,
        add_bias=add_bias,
        overwrite=1
    )

    if lam != 0.:
        XtX[np.arange(n), np.arange(n)] += lam

    cdef np.ndarray[real_t, ndim=2] XtX_copy = XtX.copy()
    cdef np.ndarray[real_t, ndim=1] XtY_copy = XtY.copy()

    cdef char lo = 'L'
    cdef int one_int = 1
    cdef int ignore = 0

    if not calc_inv:
        tposv(&lo, &n_plusb, &one_int, &XtX[0,0], &n_plusb, &XtY[0], &n_plusb, &ignore)
        return XtX_copy, XtX, XtY_copy, XtY
        ## XtX, invXtX, XtY, coef
        ##      (chol)

    else:
        tposv(&lo, &n_plusb, &one_int, &XtX[0,0], &n_plusb, &XtY[0], &n_plusb, &ignore)
        tpotri(&lo, &n_plusb, &XtX[0,0], &n_plusb, &ignore)
        return XtX_copy, XtX, XtY_copy, XtY
        ## XtX, invXtX, XtY, coef


def fit_model_inv(
        np.ndarray[real_t, ndim=2] X,
        np.ndarray[real_t, ndim=1] y,
        np.ndarray[real_t, ndim=1] w,
        Xcsr = None,
        bint add_bias=1,
        real_t lam = 1.
    ):

    cdef int n = X.shape[1] if Xcsr is None else Xcsr.shape[1]
    cdef int n_plusb = n + <int>add_bias

    cdef np.ndarray[real_t, ndim=2] invXtX = np.zeros((n_plusb,n_plusb), dtype=C_real_t)
    cdef np.ndarray[real_t, ndim=1] XtY = np.empty(n_plusb, dtype=C_real_t)

    invXtX[np.arange(n_plusb), np.arange(n_plusb)] = 1. / lam

    if np.isfortran(X):
        X = np.ascontiguousarray(X)
    if X.dtype != C_real_t:
        X = X.astype(C_real_t)

    update_matrices_inv(
        invXtX,
        XtY,
        XtY,
        X, y, w,
        Xcsr = Xcsr,
        add_bias=add_bias,
        overwrite=1
    )

    cdef real_t *ptr_XtY = &XtY[0]
    cdef real_t *ptr_invXtX = &invXtX[0,0]

    cdef np.ndarray[real_t, ndim=1] coefs = np.empty(n_plusb, dtype=C_real_t)
    cblas_tsymv(CblasRowMajor, CblasUpper, n_plusb,
                1., ptr_invXtX, n_plusb, ptr_XtY, 1,
                0., &coefs[0], 1)

    return invXtX, XtY, coefs


def update_running_noinv(
        np.ndarray[real_t, ndim=2] XtX,
        np.ndarray[real_t, ndim=1] XtY,
        np.ndarray[real_t, ndim=2] invXtX,
        np.ndarray[real_t, ndim=1] coef,
        np.ndarray[real_t, ndim=2] X,
        np.ndarray[real_t, ndim=1] y,
        np.ndarray[real_t, ndim=1] w,
        Xcsr = None,
        bint add_bias=1,
        bint calc_inv=0
    ):

    cdef int n = X.shape[1] if Xcsr is None else Xcsr.shape[1]
    cdef int n_plusb = n + <int>add_bias
    cdef int m = X.shape[0] if Xcsr is None else Xcsr.shape[0]

    cdef np.ndarray[real_t, ndim=2] Xw

    cdef char lo = 'L'
    cdef int one_int = 1
    cdef int ignore = 0

    update_matrices_noinv(
        X, y, w,
        XtX, XtY,
        Xcsr = Xcsr,
        add_bias=add_bias,
        overwrite=0
    )

    coef[:] = XtY
    invXtX[:,:] = XtX[:,:]
    tposv(&lo, &n_plusb, &one_int, &invXtX[0,0], &n_plusb, &coef[0], &n_plusb, &ignore)
    if calc_inv:
        tpotri(&lo, &n_plusb, &invXtX[0,0], &n_plusb, &ignore)

def update_running_inv(
        np.ndarray[real_t, ndim=2] invXtX,
        np.ndarray[real_t, ndim=1] XtY,
        np.ndarray[real_t, ndim=1] coefs,
        np.ndarray[real_t, ndim=1] x_vec, ### buffer
        np.ndarray[real_t, ndim=2] X,
        np.ndarray[real_t, ndim=1] y,
        np.ndarray[real_t, ndim=1] w,
        Xcsr = None,
        bint add_bias=1
    ):

    if np.isfortran(X):
        X = np.ascontiguousarray(X)

    cdef int n = X.shape[1] if Xcsr is None else Xcsr.shape[1]
    cdef int n_plusb = n + <int>add_bias
    
    update_matrices_inv(
        invXtX, XtY, x_vec,
        X, y, w,
        Xcsr = Xcsr,
        add_bias=add_bias,
        overwrite=0
    )

    cdef real_t *ptr_XtY = &XtY[0]
    cdef real_t *ptr_invXtX = &invXtX[0,0]
    cblas_tsymv(CblasRowMajor, CblasUpper, n_plusb,
                1., ptr_invXtX, n_plusb, ptr_XtY, 1,
                0., &coefs[0], 1)


def update_matrices_noinv(
        np.ndarray[real_t, ndim=2] X,
        np.ndarray[real_t, ndim=1] y,
        np.ndarray[real_t, ndim=1] w,
        np.ndarray[real_t, ndim=2] XtX,
        np.ndarray[real_t, ndim=1] XtY,
        Xcsr = None,
        bint add_bias=1,
        bint overwrite=0
    ):
    cdef int n = X.shape[1] if Xcsr is None else Xcsr.shape[1]
    cdef int n_plusb = n + <int>add_bias
    cdef int m = X.shape[0] if Xcsr is None else Xcsr.shape[0]

    cdef np.ndarray[real_t, ndim=1] Xsum
    cdef np.ndarray[real_t, ndim=2] Xw

    ### Note: in theory, can use X as either row-major or column-major,
    ### but then would have to keep track of whether XtX if upper or lower triangular,
    ### since it will produce different structures according to X.
    X = np.require(X, dtype=X.dtype, requirements=["F_CONTIGUOUS"])

    cdef CBLAS_ORDER x_ord = CblasRowMajor if not np.isfortran(X) else CblasColMajor
    if Xcsr is None:
        if w.shape[0] == 0:
            cblas_tsyrk(x_ord, CblasUpper if (x_ord==CblasRowMajor) else CblasLower, CblasTrans,
                        n, m,
                        1., &X[0,0], n if (x_ord==CblasRowMajor) else m,
                        0. if overwrite else 1., &XtX[0,0], n_plusb)
        else:
            Xw = X * w.reshape((-1,1))
            if np.isfortran(Xw) != np.isfortran(X):
                X = np.ascontiguousarray(X)
                Xw = np.ascontiguousarray(Xw)
                x_ord = CblasRowMajor
            cblas_tgemm(x_ord, CblasTrans, CblasNoTrans,
                        n, n, m,
                        1., &Xw[0,0], n if (x_ord==CblasRowMajor) else m,
                        &X[0,0], n if (x_ord==CblasRowMajor) else m,
                        0. if overwrite else 1., &XtX[0,0], n_plusb)
    else:
        if w.shape[0] == 0:
            if overwrite:
                XtX[:n,:n] = np.array(Xcsr.T.dot(Xcsr).todense())
            else:
                XtX[:n,:n] += np.array(Xcsr.T.dot(Xcsr).todense())
        else:
            Xw_sp = Xcsr.multiply(w.reshape((-1,1)))
            if overwrite:
                XtX[:n,:n] = np.array(Xw_sp.T.dot(Xcsr).todense())
            else:
                XtX[:n,:n] += np.array(Xw_sp.T.dot(Xcsr).todense())

    cdef int one_int = 1
    cdef real_t one = 1.
    if add_bias:
        if overwrite:
            XtX[n,n] = 0

        if w.shape[0] == 0:
            if Xcsr is None:
                Xsum = X.sum(axis=0)
            else:
                Xsum = np.array(Xcsr.sum(axis=0)).reshape(-1)
            XtX[n,n] += <real_t>m
        else:
            if Xcsr is None:
                Xsum = np.einsum("ij,i->j", X, w)
            else:
                Xsum = np.array(Xcsr
                                .multiply(w.reshape((-1,1)))
                                .sum(axis=1))\
                                .reshape(-1)
            XtX[n,n] += <real_t> (w.sum())
        if overwrite:
            tcopy(&n, &Xsum[0], &one_int, &XtX[0,n], &n_plusb)
        else:
            taxpy(&n, &one, &Xsum[0], &one_int, &XtX[0,n], &n_plusb)

    if XtY.shape[0]:
        if w.shape[0] == 0:
            update_XtY(
                XtY,
                X, y, np.empty(0, dtype=C_real_t),
                Xcsr = Xcsr,
                add_bias=add_bias,
                overwrite=overwrite
            )
        else:
            update_XtY(
                XtY,
                Xw if Xcsr is None else X, y, np.empty(0, dtype=C_real_t),
                Xcsr = Xw_sp if Xcsr is not None else None,
                add_bias=add_bias,
                overwrite=overwrite
            )

def update_matrices_inv(
        np.ndarray[real_t, ndim=2] invXtX,
        np.ndarray[real_t, ndim=1] XtY,
        np.ndarray[real_t, ndim=1] x_vec, ###buffer
        np.ndarray[real_t, ndim=2] X,
        np.ndarray[real_t, ndim=1] y,
        np.ndarray[real_t, ndim=1] w,
        Xcsr = None,
        bint add_bias=1,
        bint overwrite=0
    ):

    cdef int n = X.shape[1] if Xcsr is None else Xcsr.shape[1]
    cdef int n_plusb = n + <int>add_bias
    cdef int m = X.shape[0] if Xcsr is None else Xcsr.shape[0]

    if np.isfortran(X):
        X = np.ascontiguousarray(X)

    cdef real_t *prt_X = &X[0,0] if Xcsr is None else NULL
    cdef real_t *ptr_x_vec = &x_vec[0]
    cdef real_t *ptr_XtX_inv = &invXtX[0,0]
    cdef real_t *ptr_w = NULL
    cdef real_t coef
    cdef int one = 1
    cdef real_t one_real_t = 1.

    if w.shape[0]:
        ptr_w = &w[0]

    cdef real_t *ptr_csr_data = NULL
    cdef long *ptr_csr_indptr = NULL
    cdef long *ptr_csr_indices = NULL

    cdef long i
    if Xcsr is None:
        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            for i in range(m):
                cblas_tsymv(CblasRowMajor, CblasUpper, n,
                            1., ptr_XtX_inv, n_plusb, prt_X + i*n, 1,
                            0., ptr_x_vec, 1)
                if (add_bias):
                    ptr_x_vec[n] = tdot(&n, ptr_XtX_inv + n, &n_plusb, prt_X + i*n, &one)
                    ptr_x_vec[n] += ptr_XtX_inv[n_plusb*n_plusb - 1]
                    taxpy(&n, &one_real_t, ptr_XtX_inv + n, &n_plusb, ptr_x_vec, &one)

                coef = tdot(&n, ptr_x_vec, &one, prt_X + i*n, &one)
                if add_bias:
                    coef += ptr_x_vec[n]
                coef = -1. / (1. + coef)
                if ptr_w != NULL:
                    coef *= ptr_w[i]

                cblas_tsyr(CblasRowMajor, CblasUpper,
                           n_plusb, coef, ptr_x_vec, 1,
                           ptr_XtX_inv, n_plusb)

    else:
        cast_csr(Xcsr)
        ptr_csr_data = get_ptr_real_t(Xcsr.data)
        ptr_csr_indices = get_ptr_long(Xcsr.indices)
        ptr_csr_indptr = get_ptr_long(Xcsr.indptr)
        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            for i in range(m):
                tsymv_dense_sp(
                    ptr_XtX_inv, n_plusb,
                    ptr_csr_data, ptr_csr_indptr, ptr_csr_indices, i,
                    ptr_x_vec, add_bias
                )
                coef = tdot_dense_sp(ptr_x_vec, i,
                                     ptr_csr_data, ptr_csr_indptr, ptr_csr_indices)
                if add_bias:
                    coef += ptr_x_vec[n]
                coef = -1. / (1. + coef)
                if ptr_w != NULL:
                    coef *= ptr_w[i]
                cblas_tsyr(CblasRowMajor, CblasUpper,
                           n_plusb, coef, ptr_x_vec, 1,
                           ptr_XtX_inv, n_plusb)

    if ptr_w != NULL:
        y = y * w

    if XtY.shape[0]:
        update_XtY(
            XtY,
            X, y, w,
            Xcsr = Xcsr,
            add_bias=add_bias,
            overwrite=overwrite
        )


def update_XtY(
        np.ndarray[real_t, ndim=1] XtY,
        np.ndarray[real_t, ndim=2] X,
        np.ndarray[real_t, ndim=1] y,
        np.ndarray[real_t, ndim=1] w,
        Xcsr = None,
        bint add_bias=1,
        bint overwrite=0
    ):

    ## TODO: maybe this one should do the updates row-by-row,
    ## since most of the 'y' entries are expected to be zeros
    cdef CBLAS_ORDER x_ord = CblasRowMajor if not np.isfortran(X) else CblasColMajor
    
    cdef int n = X.shape[1] if Xcsr is None else Xcsr.shape[1]
    cdef int m = X.shape[0] if Xcsr is None else Xcsr.shape[0]

    cdef np.ndarray[real_t, ndim=2] Xw

    if Xcsr is None:
        if w.shape[0] == 0:
            cblas_tgemv(x_ord, CblasTrans,
                        m, n,
                        1., &X[0,0], n if (x_ord==CblasRowMajor) else m,
                        &y[0], 1,
                        0. if overwrite else 1., &XtY[0], 1)
        else:
            Xw = X * w.reshape((-1,1))
            if np.isfortran(Xw) != np.isfortran(X):
                X = np.ascontiguousarray(X)
                Xw = np.ascontiguousarray(Xw)
                x_ord = CblasRowMajor
            cblas_tgemv(x_ord, CblasTrans,
                        m, n,
                        1., &Xw[0,0], n if (x_ord==CblasRowMajor) else m,
                        &y[0], 1,
                        0. if overwrite else 1., &XtY[0], 1)
    
    else:
        if w.shape[0] == 0:
            if overwrite:
                XtY[:n] = Xcsr.T.dot(y)
            else:
                XtY[:n] += Xcsr.T.dot(y)
        else:
            Xw_sp = Xcsr.multiply(w.reshape((-1,1)))
            if overwrite:
                XtY[:n] = Xw_sp.T.dot(y)
            else:
                XtY[:n] += Xw_sp.T.dot(y)

    cdef int one_int = 1
    cdef real_t one = 1.
    if add_bias:
        if overwrite:
            XtY[n] = 0

        if w.shape[0] == 0:
            XtY[n] += y.sum()
        else:
            XtY[n] += tdot(&m, &y[0], &one_int, &w[0], &one_int)
 
def x_A_x_batch(
        np.ndarray[real_t, ndim=2] X,
        np.ndarray[real_t, ndim=2] invXtX,
        Xcsr = None,
        bint add_bias=1,
        bint copy_X=0
    ):
    
    if (X.shape[0] == 0) and (Xcsr.shape[0] == 0):
        return None
    
    cdef int n = X.shape[1] if Xcsr is None else Xcsr.shape[1]
    cdef int n_plusb = n + <int>add_bias
    cdef int m = X.shape[0] if Xcsr is None else Xcsr.shape[0]

    cdef np.ndarray[real_t, ndim=2] tempX
    cdef np.ndarray[real_t, ndim=1] outp = np.empty(m, dtype=C_real_t)

    cdef long i, j
    cdef real_t one = 1.
    cdef int one_int = 1

    cdef real_t *ptr_tempX = NULL
    cdef real_t *ptr_X = NULL
    cdef real_t *ptr_invXtX = &invXtX[0,0]
    cdef real_t *ptr_outp = &outp[0]

    if Xcsr is None:
        X = np.require(X, dtype=C_real_t, requirements=["C_CONTIGUOUS", "OWNDATA", "WRITEABLE"])

        tempX = np.zeros((m, n_plusb), dtype=C_real_t)
        ptr_tempX = &tempX[0,0]
        ptr_X = &X[0,0]

        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            cblas_tsymm(CblasRowMajor, CblasRight, CblasUpper,
                        m, n,
                        1., ptr_invXtX, n_plusb,
                        ptr_X, n,
                        0., ptr_tempX, n_plusb)
            ## TODO: this is parallelizable, but doesn't get any improvement beyond 2 threads
            for i in range(m):
                ptr_outp[i] = tdot(&n, ptr_tempX + i*n_plusb, &one_int, ptr_X + i*n, &one_int)

            if add_bias:
                cblas_tgemv(CblasRowMajor, CblasNoTrans,
                            m, n,
                            2., ptr_X, n, ptr_invXtX + n, n_plusb,
                            1., ptr_outp, 1)
                for i in range(m):
                    ptr_outp[i] += ptr_invXtX[n_plusb*n_plusb - 1]

    else:
        if Xcsr.dtype != C_real_t:
            Xcsr = Xcsr.astype(C_real_t)

        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            for i in range(n_plusb):
                for j in range(i):
                    ptr_invXtX[j + i*n_plusb] = ptr_invXtX[i + j*n_plusb]

        tempXcsr = Xcsr.dot(invXtX[:n,:n])
        outp[:] = Xcsr.multiply(tempXcsr).sum(axis=1).reshape(-1)

        if add_bias:
            outp[:] += 2. * Xcsr.dot(invXtX[n,:n])
            outp[:] += invXtX[n,n]

    return outp

def get_matrix_inv(
        np.ndarray[real_t, ndim=2] X,
        np.ndarray[real_t, ndim=2] invX
    ):
    if X.shape[0] == 0:
        return None
    cdef real_t *ptr_X = &X[0,0]
    cdef real_t *ptr_invX = &invX[0,0]
    cdef int n = X.shape[0]
    cdef int ignore = 0
    cdef char lo = 'L'
    with nogil:
        memcpy(ptr_invX, ptr_X, <size_t>(X.shape[0]*X.shape[1])*sizeof(real_t))
        tpotrf(&lo, &n, ptr_invX, &n, &ignore)
        tpotri(&lo, &n, ptr_invX, &n, &ignore)

### https://stats.stackexchange.com/questions/175271/can-i-get-a-cholesky-decomposition-from-the-inverse-of-a-matrix
### Sampling from multivariate normal N(mu, Sigma):
### - Sample Y[n,m] ~ N(0,1)
### - Convert to desired distribution by transforming:
###      A = EigVec(Sigma) * diag(sqrt(EigVal(Sigma)))
###      z = mu + Y * A
#### (If using the inverse of Sigma, the eigen vectors are the same,
####  while the eigen values are 1 divided over the eigen values of the inverse)
def get_mvnorm_multiplier(
        np.ndarray[real_t, ndim=2] Sigma,
        real_t multiplier,
        bint is_inverse,
        bint overwrite_sigma
    ):

    Sigma = np.require(Sigma, dtype=C_real_t, requirements=["C_CONTIGUOUS", "OWNDATA"])
    cdef int m = Sigma.shape[0]
    cdef np.ndarray[real_t, ndim=2] eig_Vecs = np.empty((0,0), dtype=C_real_t)
    if overwrite_sigma:
        eig_Vecs = Sigma
    else:
        eig_Vecs = Sigma.copy()
    cdef np.ndarray[real_t, ndim=1] eig_vals = np.empty(m, dtype=C_real_t)
    cdef real_t *ptr_Ev = &eig_Vecs[0,0]
    cdef real_t *ptr_ev = &eig_vals[0]

    cdef char lo = 'L'
    cdef int one = 1
    cdef int minus_one = -1
    cdef char V = 'V'
    cdef int ignore
    cdef real_t size_buffer

    tsyev(&V, &lo, &m, ptr_Ev, &m, ptr_ev, &size_buffer, &minus_one, &ignore)
    cdef int size_buffer_as_int = int(size_buffer)
    cdef np.ndarray[real_t, ndim=1] buffer_arr = np.empty(size_buffer_as_int, dtype=C_real_t)
    tsyev(&V, &lo, &m, ptr_Ev, &m, ptr_ev, &buffer_arr[0], &size_buffer_as_int, &ignore)

    ### Note: the obtained eigen values are transposed at this point as they were passed
    ### assuming column-major (input was a symmetric matrix so it doesn't matter).
    ### If doing the rest of this part in C, would have to use tscal with incX given by
    ### the number of columns

    ### Note2: If multiplying the input matrix by a constant, the eigen vectors
    ### remain the same, while the eigen values get multiplied by that same constant
    if is_inverse:
        ix_replace = eig_vals <= 1e-10
        eig_vals[ix_replace] = 1.
        eig_vals[:] =  1. / eig_vals
        eig_vals[ix_replace] = 0.
    eig_vals[:] = eig_vals.clip(min=0.)
    eig_vals_orig = eig_vals.copy()
    eig_vals[:] =  np.sqrt(multiplier * eig_vals)

    return np.einsum("ij,i->ij", eig_Vecs, eig_vals), eig_vals, eig_vals_orig


def mvnorm_from_Eig(
        np.ndarray[real_t, ndim=1] mu,
        np.ndarray[real_t, ndim=2] EigMultiplier,
        size_t n,
        rng
    ):
    cdef size_t m = EigMultiplier.shape[0]
    cdef np.ndarray[real_t, ndim=2] X = np.empty((0,0), dtype=C_real_t)
    if isinstance(rng, np.random.Generator):
        X = rng.standard_normal(size=(n, m), dtype=C_real_t)
    else:
        X = rng.normal(size=(n, m), dtype=C_real_t).astype(C_real_t)
    return mu + X.dot(EigMultiplier)


def mvnorm_from_Eig_different_m(
        np.ndarray[real_t, ndim=1] mu,
        np.ndarray[real_t, ndim=2] EigMultiplier,
        np.ndarray[real_t, ndim=1] eig_vals_multiplied,
        np.ndarray[real_t, ndim=1] eig_vals_orig,
        real_t new_m,
        size_t n,
        rng,
        bint return_multiplier
    ):
    new_EigMultiplier = EigMultiplier / eig_vals_multiplied.reshape((-1,1))
    new_EigMultiplier = np.einsum("ij,i->ij",
                                  new_EigMultiplier,
                                  np.sqrt(new_m * eig_vals_orig))
    if return_multiplier:
        return new_EigMultiplier, np.sqrt(new_m * eig_vals_orig)
    else:
        return mvnorm_from_Eig(mu, new_EigMultiplier, n, rng)

## https://stats.stackexchange.com/questions/175271/can-i-get-a-cholesky-decomposition-from-the-inverse-of-a-matrix
def mvnorm_from_invcov(
        np.ndarray[real_t, ndim=1] mu,
        np.ndarray[real_t, ndim=2] invSigma,
        size_t size,
        rng
    ):
    
    ### First calculate eigen values and eigen vectors
    cdef int m = invSigma.shape[0]
    cdef np.ndarray[real_t, ndim=2] eig_Vecs = invSigma.copy()
    cdef np.ndarray[real_t, ndim=1] eig_vals = np.empty(m, dtype=C_real_t)
    cdef real_t *ptr_Ev = &eig_Vecs[0,0]
    cdef real_t *ptr_ev = &eig_vals[0]

    cdef char lo = 'L'
    cdef int one = 1
    cdef int minus_one = -1
    cdef char V = 'V'
    cdef int ignore
    cdef real_t size_buffer

    tsyev(&V, &lo, &m, ptr_Ev, &m, ptr_ev, &size_buffer, &minus_one, &ignore)
    cdef int size_buffer_as_int = int(size_buffer)
    cdef np.ndarray[real_t, ndim=1] buffer_arr = np.empty(size_buffer_as_int, dtype=C_real_t)
    tsyev(&V, &lo, &m, ptr_Ev, &m, ptr_ev, &buffer_arr[0], &size_buffer_as_int, &ignore)

    ### Note: the obtained eigen values are transposed at this point as they were passed
    ### assuming column-major (input was a symmetric matrix so it doesn't matter).
    ### If doing the rest of this part in C, would have to use tscal with incX given by
    ### the number of columns

    ### Now use the formula
    cdef size_t n = mu.shape[0]
    if isinstance(rng, np.random.Generator):
        sample_stdnorm = rng.standard_normal(size=(n, m), dtype=C_real_t)
    else:
        sample_stdnorm = rng.normal(size=(n, m)).astype(C_real_t)
    return mu + np.dot(
        sample_stdnorm,
        np.einsum("ij,i->ij", eig_Vecs, np.sqrt(1. / eig_vals))
    )

def add_to_diag(np.ndarray[real_t, ndim=2] XtX, bint add_bias, real_t lam):
    if not XtX.shape[0]:
        return None
    cdef size_t n = XtX.shape[0] - add_bias
    cdef size_t lda = XtX.shape[0]
    cdef size_t ix
    cdef real_t *ptr_XtX = &XtX[0,0]
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        for ix in range(n):
            ptr_XtX[ix + ix*lda] += lam

def fill_lower_triangle(np.ndarray[real_t, ndim=2] XtX):
    if not XtX.shape[0]:
        return None
    cdef size_t n = XtX.shape[0]
    cdef size_t ix, row, col
    cdef real_t *ptr_XtX = &XtX[0,0]
    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        for row in range(1, n):
            for col in range(row):
                ptr_XtX[col + row*n] = ptr_XtX[row + col*n]

def solve_elasticnet(
        np.ndarray[real_t, ndim=2] XtX,
        np.ndarray[real_t, ndim=1] XtY2,
        np.ndarray[real_t, ndim=1] prev_coef,
        np.ndarray[real_t, ndim=1] coef,
        bint add_bias=1
    ):
    
    cdef int n = XtX.shape[0]
    cdef int ix
    cdef real_t *ptr_XtX = &XtX[0,0]
    cdef real_t *ptr_XtY_pos = &XtY2[0]
    cdef real_t *ptr_XtY_neg = ptr_XtY_pos + n
    cdef real_t *ptr_prev_pos = &prev_coef[0]
    cdef real_t *ptr_prev_neg = ptr_prev_pos + n
    cdef int one = 1

    cdef real_t diff_iter = 0.
    cdef real_t diff_val = 0.
    cdef real_t newval = 0

    with nogil, boundscheck(False), nonecheck(False), wraparound(False):
        while True:
            diff_iter = 0;
            for ix in range(n):
                newval = ptr_prev_pos[ix] + ptr_XtY_pos[ix] / ptr_XtX[ix + ix*n]
                newval = newval if newval >= 0. else 0.
                diff_val = newval - ptr_prev_pos[ix]
                if (fabs_t(diff_val) > 1e-10):
                    diff_iter += fabs_t(diff_val)
                    taxpy(&n, &diff_val, ptr_XtX + ix*n, &one, ptr_XtY_neg, &one)
                    diff_val = -diff_val
                    taxpy(&n, &diff_val, ptr_XtX + ix*n, &one, ptr_XtY_pos, &one)
                    ptr_prev_pos[ix] = newval;
            for ix in range(n):
                newval = ptr_prev_neg[ix] + ptr_XtY_neg[ix] / ptr_XtX[ix + ix*n]
                newval = newval if newval >= 0. else 0.
                diff_val = newval - ptr_prev_neg[ix]
                if fabs_t(diff_val) > 1e-10:
                    diff_iter += fabs_t(diff_val)
                    taxpy(&n, &diff_val, ptr_XtX + ix*n, &one, ptr_XtY_pos, &one)
                    diff_val = -diff_val
                    taxpy(&n, &diff_val, ptr_XtX + ix*n, &one, ptr_XtY_neg, &one)
                    ptr_prev_neg[ix] = newval
            
            if (isnan(diff_iter)) or (not isfinite(diff_iter)) or (diff_iter < 1e-8):
                break
        for ix in range(n):
            coef[ix] = ptr_prev_pos[ix] - ptr_prev_neg[ix]
