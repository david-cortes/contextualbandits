import numpy as np
import ctypes
from scipy.sparse import issparse, isspmatrix_csr
import warnings
from sklearn.base import BaseEstimator
from . import _wrapper_double, _wrapper_float

__all__ = ["LinearRegression", "ElasticNet"]

def _is_row_major(X):
    if (X is None) or (issparse(X)):
        return True
    else:
        return X.strides[1] == X.dtype.itemsize

def _copy_if_subview(X):
    ### TODO: the Cython functions should accept a 'leading dimension'
    ### parameter so as to avoid copying the data here
    if (X is not None) and (not issparse(X)):
        row_major = _is_row_major(X)
        leading_dimension = int(X.strides[0 if row_major else 1] / X.dtype.itemsize)
        if leading_dimension != X.shape[1 if row_major else 0]:
            X = X.copy()
        elif (len(X.strides) != 2) or (not X.flags.aligned):
            X = X.copy()
        if _is_row_major(X) != row_major:
            X = np.ascontiguousarray(X)
    return X

def _copy_if_subview_1d(arr):
    if arr is not None:
        if not isinstance(arr, np.ndarray):
            arr = np.ascontiguousarray(arr)
        arr = arr.reshape(-1)
        if (arr.strides[0] != arr.dtype.itemsize) or (not arr.flags.aligned):
            arr = np.ascontiguousarray(arr)
    return arr

class LinearRegression(BaseEstimator):
    """
    Linear Regression
    
    Typical Linear Regression model, which keeps track of the aggregated data
    needed to obtain the closed-form solution in a way that calling 'partial_fit'
    multiple times would be equivalent to a single call to 'fit' with all the data.
    This is an exact method rather than a stochastic optimization procedure.

    Also provides functionality for making predictions according to upper confidence
    bound (UCB) and to Thompson sampling criteria.

    Note
    ----
    Doing linear regression this way requires both memory and computation time
    which scale quadratically with the number of columns/features/variables. As
    such, the class will by default use C 'float' types (typically ``np.float32``)
    instead of C 'double' (``np.float64``), in order to save memory.

    Parameters
    ----------
    lambda_ : float
        Strenght of the L2 regularization.
    fit_intercept : bool
        Whether to add an intercept term to the formula. If passing 'True', it will
        be the last entry in the coefficients.
    method : str, one of 'chol' or 'sm'
        Method used to fit the model. Options are:

        ``'chol'``:
            Uses the Cholesky decomposition to solve the linear system from the
            least-squares closed-form each time 'fit' or 'partial_fit' is called.
            This is likely to be faster when fitting the model to a large number
            of observations at once, and is able to better exploit multi-threading.
        ``'sm'``:
            Starts with an inverse diagonal matrix and updates it as each
            new observation comes using the Sherman-Morrison formula, thus
            never explicitly solving the linear system, nor needing to calculate
            a matrix inverse. This is likely to be faster when fitting the model
            to small batches of observations. Be aware that with this method, it
            will add regularization to the intercept if passing 'fit_intercept=True'.

        Note that it is possible to change the method after the object has
        already been fit (e.g. if you want non-regularization intercept
        with fast online updates, you might use Cholesky first and then switch
        to Sherman-Morrison).
    calc_inv : bool
        When using ``method='chol'``, whether to also produce a matrix inverse, which
        is required for using the LinUCB prediction mode. Ignored when
        passing ``method='sm'`` (the default). Note that is is possible to change
        the method after the object has already been fit.
    precompute_ts : bool
        Whether to pre-compute the necessary matrices to accelerate the Thompson
        sampling prediction mode (method ``predict_thompson``). If you plan to use
        ``predict_thompson``, it's recommended to pass "True".
        Note that this will make the Sherman-Morrison updates (``method="sm"``)
        much slower as it will calculate eigenvalues after every update.
        Can be changed after the object is already initialized or fitted.
    precompute_ts_multiplier : float
        Multiplier for the covariance matrix to use when using ``precompute_ts``.
        Calling ``predict_thompson`` with this same multiplier will be faster than
        with a different one. Calling it with a different multiplier with
        ``precompute_ts`` will still be faster than without it, unless using
        also ``n_presampled``.
        Ignored when passing ``precompute_ts=False``.
    n_presampled : None or int
        When passing ``precompute_ts``, this denotes a number of coefficients to pre-sample
        after calling 'fit' and/or 'partial_fit', which will be used later
        when calling ``predict_thompson`` with the same multiplier as in ``precompute_ts_multiplier``.
        Pre-sampling a large number of coefficients can help to speed up Thompson-sampled predictions
        at the expense of longer fitting times, and is recommended if there is a large number of
        predictions between calls to 'fit' or 'partial_fit'.
        If passing 'None' (the default), will not pre-sample a finite number of the coefficients
        at fitting time, but will rather sample (different) coefficients in calls to
        ``predict_thompson``.
        The pre-sampled coefficients will not be used if calling ``predict_thompson`` with
        a different multiplier than what was passed to ``precompute_ts_multiplier``.
    rng_presample : None, int, RandomState, or Generator
        Random number generator to use for pre-sampling coefficients.
        If passing an integer, will use it as a random seed for initialization. If passing
        a RandomState, will use it to draw an integer to use as seed. If passing a
        Generator, will use it directly. If passing 'None', will initialize a Generator
        without random seed.
        Ignored if passing ``precompute_ts=False`` or ``n_presampled=None`` (the defaults).
    use_float : bool
        Whether to use C 'float' type for the required matrices. If passing 'False',
        will use C 'double'. Be aware that memory usage for this model can grow
        very large. Can be changed after initialization.

    Attributes
    ----------
    coef_ : array(n) or array(n+1)
        The obtained coefficients. If passing 'fit_intercept=True', the intercept
        will be at the last entry.
    """
    def __init__(self, lambda_=1., fit_intercept=True, method="sm",
                 calc_inv=True, precompute_ts=False,
                 precompute_ts_multiplier=1.,
                 n_presampled=None, rng_presample=None,
                 use_float=True):
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
        self._method = method
        self._calc_inv = bool(calc_inv)
        self._precompute_ts = bool(precompute_ts)
        self._precompute_ts_multiplier = precompute_ts_multiplier
        self._n_presampled = n_presampled
        self.rng_presample = rng_presample
        self._use_float = bool(use_float)

        self._set_dtype(force_cast=False)
        self.coef_ = np.empty(0, dtype=self._dtype)
        self._XtX = np.empty((0,0), dtype=self._dtype)
        self._invXtX = np.empty((0,0), dtype=self._dtype)
        self._XtY = np.empty(0, dtype=self._dtype)
        self._bufferX = np.empty(0, dtype=self._dtype)
        self._n = 0

        ### These are only used alongside precomputed TS
        self._EigMultiplier = np.empty((0,0), dtype=self._dtype)
        self._EigValsUsed = np.empty(0, dtype=self._dtype)
        self._EigValsOrig = np.empty(0, dtype=self._dtype)
        self._coef_precomputed = np.empty((0,0), dtype=self._dtype)

        self.is_fitted_ = False

    def _set_dtype(self, force_cast=False):
        self._dtype = ctypes.c_float if self._use_float else ctypes.c_double

        if force_cast:
            if self.coef_.dtype != self._dtype:
                self.coef_ = self.coef_.astype(self._dtype)
            if self._XtX.dtype != self._dtype:
                self._XtX = self._XtX.astype(self._dtype)
            if self._invXtX.dtype != self._dtype:
                self._invXtX = self._invXtX.astype(self._dtype)
            if self._XtY.dtype != self._dtype:
                self._XtY = self._XtY.astype(self._dtype)
            if self._bufferX.dtype != self._dtype:
                self._bufferX = self._bufferX.astype(self._dtype)
            if self._EigMultiplier.dtype != self._dtype:
                self._EigMultiplier = self._EigMultiplier.astype(self._dtype)
            if self._EigValsUsed.dtype != self._dtype:
                self._EigValsUsed = self._EigValsUsed.astype(self._dtype)
            if self._EigValsOrig.dtype != self._dtype:
                self._EigValsOrig = self._EigValsOrig.astype(self._dtype)
            if self._coef_precomputed.dtype != self._dtype:
                self._coef_precomputed = self._coef_precomputed.astype(self._dtype)

    @property
    def use_float(self):
        return self._use_float
    @use_float.setter
    def use_float(self, value):
        self._use_float = use_float
        self._set_dtype(force_cast=True)

    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self, value):
        assert value in ["chol", "sm"]
        if (self._method != value) and (self.is_fitted_):
            cy_funs = _wrapper_float if self._use_float else _wrapper_double
            if self._method == "sm":
                self._XtX = np.empty(self._invXtX.shape, dtype=self._dtype)
                cy_funs.get_matrix_inv(self._invXtX, self._XtX)
            elif (not self._calc_inv):
                self._invXtX = np.empty(self._XtX.shape, dtype=self._dtype)
                cy_funs.get_matrix_inv(self._XtX, self._invXtX)
                self._calc_inv = True
        self._method = value

    @property
    def calc_inv(self):
        return self._calc_inv
    
    @calc_inv.setter
    def calc_inv(self, value):
        if bool(value) != bool(self._calc_inv):
            if self._method == "chol":
                self._invXtX = np.empty(self._XtX.shape, dtype=self._dtype)
                if self.is_fitted_:
                    cy_funs = _wrapper_float if self._use_float else _wrapper_double
                    cy_funs.get_matrix_inv(self._XtX, self._invXtX)
            self._calc_inv = bool(value)

    @property
    def precompute_ts(self):
        return self._precompute_ts
    
    @precompute_ts.setter
    def precompute_ts(self, value):
        if (bool(value) != bool(self._precompute_ts)) and (self.is_fitted_):
            ### Do not set the value right away as the procedure can
            ### be interrupted in the middle and would leave the object
            ### in an unusable state
            if not value:
                self._precompute_ts = value
                self._EigMultiplier = np.empty((0,0), dtype=self._dtype)
                self._EigValsUsed = np.empty(0, dtype=self._dtype)
                self._EigValsOrig = np.empty(0, dtype=self._dtype)
                self._coef_precomputed = np.empty((0,0), dtype=self._dtype)
            else:
                assert self.precompute_ts_multiplier > 0.
                cy_funs = _wrapper_float if self._use_float else _wrapper_double
                self._EigMultiplier, self._EigValsUsed, self._EigValsOrig = \
                    cy_funs.get_mvnorm_multiplier(self._XtX
                                                    if (self.method == "chol")
                                                    else self._invXtX,
                                                  self.precompute_ts_multiplier,
                                                  self.method != "chol", False)
                if self.n_presampled is not None:
                    self._presample()
                self._precompute_ts = value
        else:
            self._precompute_ts = value

    @property
    def precompute_ts_multiplier(self):
        return float(self._precompute_ts_multiplier)
    
    @precompute_ts_multiplier.setter
    def precompute_ts_multiplier(self, value):
        if (not self._precompute_ts) or (not self.is_fitted_):
            self._precompute_ts_multiplier = value
        elif value != self._precompute_ts_multiplier:
            assert value > 0.
            if not isinstance(value, float):
                value = float(value)
            cy_funs = _wrapper_float if self._use_float else _wrapper_double
            self._EigMultiplier, self._EigValsUsed = \
                cy_funs.mvnorm_from_Eig_different_m(np.empty(0, dtype=self._dtype),
                                                    self._EigMultiplier,
                                                    self._EigValsUsed,
                                                    self._EigValsOrig,
                                                    value, 0, np.random,
                                                    True)
            if self.n_presampled is not None:
                self._presample()
            self._precompute_ts_multiplier = value

    @property
    def n_presampled(self):
        return self._n_presampled
    
    @n_presampled.setter
    def n_presampled(self, value):
        if value != self.n_presampled:
            if value is None:
                self._coef_precomputed = np.empty((0,0), dtype=self._dtype)
            else:
                if (self.is_fitted_) and (self._precompute_ts):
                    self._presample()
        self.n_presampled = n_presampled

    def _set_rng(self):
        if isinstance(self.rng_presample, np.random.RandomState):
            self.rng_presample = self.rng_presample.randint(0, np.iinfo(np.int32).max)
        if isinstance(self.rng_presample, float):
            self.rng_presample = int(self.rng_presample)
        if self.rng_presample is None:
            self.rng_presample = np.random.Generator(np.random.MT19937())
        elif isinstance(self.rng_presample, int):
            self.rng_presample = np.random.Generator(np.random.MT19937(seed = self.rng_presample))

    def _presample(self):
        self._set_rng()
        cy_funs = _wrapper_float if self._use_float else _wrapper_double
        self._coef_precomputed = \
            cy_funs.mvnorm_from_Eig(self.coef_,
                                    self._EigMultiplier,
                                    self._n_presampled,
                                    self.rng_presample)

    def _process_X_y_w(self, X, y, sample_weight, only_X=False):
        if X.dtype != self._dtype:
            X = X.astype(self._dtype)
        if not issparse(X):
            Xcsr = None
            if X.__class__.__name__ == "DataFrame":
                X = X.to_numpy()
            if (not isinstance(X, np.ndarray)) or (not _is_row_major(X)):
                X = np.ascontiguousarray(X)
            if len(X.shape) != 2:
                raise ValueError("'X' must be 2-dimensional")
            X = _copy_if_subview(X)
        else:
            if not isspmatrix_csr(X):
                warnings.warn("Sparse matrices only supported in CSR format. Input will be converted.")
                X = csr_matrix(X)
            Xcsr = X
            X = np.empty((0,0), dtype=self._dtype)
            if len(Xcsr.shape) != 2:
                raise ValueError("'X' must be 2-dimensional")
        if only_X:
            return X, Xcsr

        if issparse(y):
            raise ValueError("Sparse 'y' not supported.")
        if issparse(sample_weight):
            raise ValueError("Sparse 'sample_weight' not supported.")

        y = np.array(y).reshape(-1)
        if sample_weight is None:
            w = np.empty(0, dtype=self._dtype)
        else:
            w = sample_weight

        y = _copy_if_subview_1d(y)
        w = _copy_if_subview_1d(w)

        if Xcsr is None:
            assert X.shape[0] == y.shape[0]
        else:
            if Xcsr.shape[0] != y.shape[0]:
                raise ValueError("'X' and 'y' must have the same number of rows.")

        if sample_weight is not None:
            if X.shape[0] != w.shape[0]:
                raise ValueError("'sample_weight' must have the same number of rows as 'X' and 'y'.")

        if y.dtype != self._dtype:
            y = y.astype(self._dtype)
        if w.dtype != self._dtype:
            w = w.astype(self._dtype)

        return X, Xcsr, y, w

    def fit(self, X, y, sample_weight=None):
        """
        Fit model to data

        Note
        ----
        Calling 'fit' will reset whatever previous data was there. For fitting
        the model incrementally to new data, use 'partial_fit' instead.

        Parameters
        ----------
        X : array(m,n) or CSR matrix(m, n)
            The covariates.
        y : array-like(m)
            The target variable.
        sample_weight : None or array-like(m)
            Observation weights for each row.

        Returns
        -------
        self

        """
        assert self.method in ["chol", "sm"]
        assert self.precompute_ts_multiplier > 0

        self._n = X.shape[1]
        X, Xcsr, y, w = self._process_X_y_w(X, y, sample_weight)

        cy_funs = _wrapper_float if self._use_float else _wrapper_double
        if (self.method == "chol") or (X.shape[0] >= X.shape[1]):
            self._XtX, self._invXtX, self._XtY, self.coef_ = \
                cy_funs.fit_model_noinv(
                    X, y, w, Xcsr,
                    add_bias=self.fit_intercept,
                    lam = self.lambda_,
                    calc_inv=(self.calc_inv) or (self.method != "chol")
                )
            if self._precompute_ts:
                self._EigMultiplier, self._EigValsUsed, self._EigValsOrig = \
                    cy_funs.get_mvnorm_multiplier(self._XtX,
                                                  self.precompute_ts_multiplier,
                                                  False, self.method != "chol")
                if self.n_presampled is not None:
                    self._presample()
        else:
            self._invXtX, self._XtY, self.coef_ = \
                cy_funs.fit_model_inv(
                    X, y, w, Xcsr,
                    add_bias=self.fit_intercept,
                    lam = self.lambda_
                )
            if self._precompute_ts:
                self._EigMultiplier, self._EigValsUsed, self._EigValsOrig = \
                    cy_funs.get_mvnorm_multiplier(self._invXtX,
                                                  self.precompute_ts_multiplier,
                                                  True, False)
                if self.n_presampled is not None:
                    self._presample()

        if self.method != "chol":
            self._bufferX = np.empty(self._n + int(self.fit_intercept), dtype=self._dtype)

        self.is_fitted_ = True
        return self


    def partial_fit(self, X, y, sample_weight=None, *args, **kwargs):
        """
        Fit model incrementally to new data

        Parameters
        ----------
        X : array(m,n) or CSR matrix(m, n)
            The covariates.
        y : array-like(m)
            The target variable.
        sample_weight : None or array-like(m)
            Observation weights for each row.

        Returns
        -------
        self

        """
        if not self.is_fitted_:
            return self.fit(X, y, sample_weight)

        if X.shape[1] != self._n:
            raise ValueError("Number of columns in X doesn't match with previous.")
        X, Xcsr, y, w = self._process_X_y_w(X, y, sample_weight)

        cy_funs = _wrapper_float if self._use_float else _wrapper_double
        if self.method == "chol":
            if self._invXtX.shape[0] == 0:
                self._invXtX = np.empty((self._XtX.shape[0], self._XtX.shape[1]),
                                        dtype=ctypes.c_float if self._use_float else ctypes.c_double)
            cy_funs.update_running_noinv(
                self._XtX, self._XtY, self._invXtX, self.coef_,
                X, y, w, Xcsr,
                add_bias=self.fit_intercept,
                calc_inv=self.calc_inv
            )
            if self._precompute_ts:
                self._EigMultiplier, self._EigValsUsed, self._EigValsOrig = \
                    cy_funs.get_mvnorm_multiplier(self._XtX,
                                                  self.precompute_ts_multiplier,
                                                  False, False)
                if self.n_presampled is not None:
                    self._presample()
        else:
            if not self._bufferX.shape[0]:
                self._bufferX = np.empty(self._n + int(self.fit_intercept), dtype=self._dtype)
            cy_funs.update_running_inv(
                self._invXtX,
                self._XtY,
                self.coef_,
                self._bufferX,
                X, y, w, Xcsr,
                add_bias=self.fit_intercept,
            )
            if self._precompute_ts:
                self._EigMultiplier, self._EigValsUsed, self._EigValsOrig = \
                    cy_funs.get_mvnorm_multiplier(self._invXtX,
                                                  self.precompute_ts_multiplier,
                                                  True, False)
                if self.n_presampled is not None:
                    self._presample()
        return self

    def predict(self, X):
        """
        Make predictions on new data

        Parameters
        ----------
        X : array(m,n) or CSR matrix(m, n)
            The covariates.

        Returns
        -------
        y_hat : array(m)
            The predicted values given 'X'.
        """
        assert self.is_fitted_

        pred = X.dot(self.coef_[:self._n])
        if self.fit_intercept:
            pred[:] += self.coef_[self._n]
        return pred

    def _multiply_x_A_x(self, X):
        X, Xcsr = self._process_X_y_w(X, None, None, only_X=True)
        cy_funs = _wrapper_float if self._use_float else _wrapper_double
        return cy_funs.x_A_x_batch(X, self._invXtX, Xcsr, self.fit_intercept)

    def predict_ucb(self, X, alpha=1.0, add_unfit_noise=False, random_state=None):
        """
        Make an upper-bound prediction on new data

        Make a prediction on new data with an upper bound given by the LinUCB
        formula (be aware that it's not probabilistic like a regular CI).

        Note
        ----
        If using this method, it's recommended to center the 'X' data passed
        to 'fit' and 'partial_fit'. If not centered, it's recommendable to
        lower the ``alpha`` value.

        Parameters
        ----------
        X : array(m,n) or CSR matrix(m, n)
            The covariates.
        alpha : float > 0 or array(m, ) > 0
            The multiplier for the width of the bound. Can also pass an array
            with different values for each row.
        add_unfit_noise : bool
            When making predictions with an unfit model (in this case they are
            given by empty zero matrices except for the inverse diagonal matrix
            based on the regularization parameter), whether to add a very small
            amount of random noise ~ Uniform(0, 10^-12) to it. This is useful in
            order to break ties at random when using multiple models.
        random_state : None, np.random.Generator, or np.random.RandomState
            A NumPy 'Generator' or 'RandomState' object instance to use for generating
            random numbers. If passing 'None', will use NumPy's random
            module directly (which can be made reproducible through
            ``np.random.seed``). Only used when passing ``add_unfit_noise=True``
            and calling this method on a model that has not been fit to data.

        Returns
        -------
        y_hat : array(m)
            The predicted upper bound on 'y' given 'X' and ``alpha``.

        References
        ----------
        .. [1] Chu, Wei, et al. "Contextual bandits with linear payoff functions."
               Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. 2011.
        """
        if not isinstance(alpha, np.ndarray):
            assert alpha > 0.
            if isinstance(alpha, int):
                alpha = float(alpha)
            assert isinstance(alpha, float)
        else:
            assert alpha.shape[0] == X.shape[0]

        if (self.method == "chol") and (not self.calc_inv):
            raise ValueError("Not available when using 'method=\"chol\"' and 'calc_inv=False'.")

        if not self.is_fitted_:
            pred = alpha * np.sqrt(np.einsum("ij,ij->i", X, X) / self.lambda_)
            if add_unfit_noise:
                if random_state is None:
                    noise = np.random.uniform(low=0., high=1e-12, size=X.shape[0])
                else:
                    noise = random_state.uniform(low=0., high=1e-12, size=X.shape[0])
                pred[:] += noise
            return pred

        pred = self.predict(X)
        ci = self._multiply_x_A_x(X)
        pred[:] += alpha * np.sqrt(ci)
        return pred

    def predict_thompson(self, X, v_sq=1.0, sample_unique=False, random_state=None):
        """
        Make a guess prediction on new data

        Make a prediction on new data with coefficients sampled from their
        estimated distribution.

        Note
        ----
        If using this method, it's recommended to center the 'X' data passed
        to 'fit' and 'partial_fit'. If not centered, it's recommendable to
        lower the ``v_sq`` value.

        Parameters
        ----------
        X : array(m,n) or CSR matrix(m, n)
            The covariates.
        v_sq : float > 0
            The multiplier for the covariance matrix. Larger values lead to
            more variable results.
        sample_unique : bool
            Whether to sample different coefficients each time a prediction is to
            be made. If passing 'False', when calling 'predict', it will sample
            the same coefficients for all the observations in the same call to
            'predict', whereas if passing 'True', will use a different set of
            coefficients for each observations. Passing 'False' leads to an
            approach which is theoretically wrong, but as sampling coefficients
            can be very slow, using 'False' can provide a reasonable speed up
            without much of a performance penalty.
        random_state : None, np.random.Generator, or np.random.RandomState
            A NumPy 'Generator' or 'RandomState' object instance to use for generating
            random numbers. If passing 'None', will use NumPy's random
            module directly (which can be made reproducible through
            ``np.random.seed``).

        Returns
        -------
        y_hat : array(m)
            The predicted guess on 'y' given 'X' and ``v_sq``.

        References
        ----------
        .. [1] Agrawal, Shipra, and Navin Goyal.
               "Thompson sampling for contextual bandits with linear payoffs."
               International Conference on Machine Learning. 2013.
        """
        assert self.is_fitted_
        if X.shape[1] != (self.coef_.shape[0]-int(self.fit_intercept)):
            raise ValueError("'X' has wrong number of columns.")
        assert v_sq > 0.
        if isinstance(v_sq, int):
            v_sq = float(v_sq)
        assert isinstance(v_sq, float)

        cy_funs = _wrapper_float if self._use_float else _wrapper_double
        random_state = np.random if random_state is None else random_state

        if not self._precompute_ts:
            tol = 1e-15

            if self.method != "chol":
                if np.linalg.det(self._invXtX) >= tol:
                    inv_cov = self._invXtX
                else:
                    inv_cov = self._invXtX.copy()
                    n = inv_cov.shape[1]
                    for i in range(10):
                        inv_cov[np.arange(n), np.arange(n)] += 1e-1
                        if np.linalg.det(inv_cov) >= tol:
                            break
                    np.nan_to_num(inv_cov, copy=False)
            else:
                if np.linalg.det(self._XtX) >= tol:
                    cov = self._XtX
                else:
                    cov = self._XtX.copy()
                    n = cov.shape[1]
                    for i in range(10):
                        cov[np.arange(n), np.arange(n)] += 1e-1
                        if np.linalg.det(cov) >= tol:
                            break
                    np.nan_to_num(cov, copy=False)



        if (self.n_presampled is not None) and (np.abs(v_sq - self.precompute_ts_multiplier) <= 1e-3):
            self._set_rng()
            ix_take = self.rng_presample.integers(self.n_presampled, size=X.shape[0], replace=True)
            coef = self._coef_precomputed[ix_take]
            if not issparse(X):
                pred = np.einsum("ij,ij->i", X, coef[:, :X.shape[1]])
            else:
                pred = X.multiply(coef[:, :X.shape[1]]).sum(axis=1)
            if self.fit_intercept:
                pred[:] += coef[:, -1]
        elif sample_unique:
            if self._precompute_ts:
                if np.abs(v_sq - self.precompute_ts_multiplier) <= 1e-3:
                    coef = cy_funs.mvnorm_from_Eig(self.coef_,
                                                   self._EigMultiplier,
                                                   X.shape[0],
                                                   random_state)
                else:
                    coef = cy_funs.mvnorm_from_Eig_different_m(self.coef_,
                                                               self._EigMultiplier,
                                                               self._EigValsUsed,
                                                               self._EigValsOrig,
                                                               v_sq,
                                                               X.shape[0],
                                                               random_state,
                                                               False)
            elif self.method == "chol":
                coef = random_state.multivariate_normal(mean=self.coef_,
                                                        cov=v_sq * cov,
                                                        size=X.shape[0],
                                                        method="cholesky")
            else:
                coef = cy_funs.mvnorm_from_invcov(self.coef_,
                                                  (1./v_sq) * inv_cov,
                                                  size=X.shape[0],
                                                  rng=random_state)
            if not issparse(X):
                pred = np.einsum("ij,ij->i", X, coef[:, :X.shape[1]])
            else:
                pred = X.multiply(coef[:, :X.shape[1]]).sum(axis=1)
            if self.fit_intercept:
                pred[:] += coef[:, -1]
        else:
            if self._precompute_ts:
                if np.abs(v_sq - self.precompute_ts_multiplier) <= 1e-3:
                    coef = cy_funs.mvnorm_from_Eig(self.coef_,
                                                   self._EigMultiplier,
                                                   1,
                                                   random_state)
                else:
                    coef = cy_funs.mvnorm_from_Eig_different_m(self.coef_,
                                                               self._EigMultiplier,
                                                               self._EigValsUsed,
                                                               self._EigValsOrig,
                                                               v_sq,
                                                               1,
                                                               random_state,
                                                               False)
            elif self.method == "chol":
                coef = random_state.multivariate_normal(mean=self.coef_,
                                                        cov=v_sq * cov,
                                                        method="cholesky")
            else:
                coef = cy_funs.mvnorm_from_invcov(self.coef_,
                                                  (1./v_sq) * inv_cov,
                                                  size=1,
                                                  rng=random_state)
            coef = coef.reshape(-1)
            pred = X.dot(coef[:X.shape[1]])
            if self.fit_intercept:
                pred[:] += coef[-1]

        return pred

class ElasticNet(BaseEstimator):
    """
    ElasticNet Regression

    ElasticNet regression (with penalization on the l1 and l2 norms of
    the coefficients),  which keeps track of the aggregated data
    needed to obtain the optimal coefficients in such a way that calling 'partial_fit'
    multiple times would be equivalent to a single call to 'fit' with all the data.
    This is an exact method rather than a stochastic optimization procedure.
    
    Note
    ----
    This ElasticNet regression is fit through a reduction to non-negative
    least squares with twice the number of variables, which is in turn solved
    through a coordinate descent procedure. This is typically slower than the
    lasso paths used in GLMNet and SciKit-Learn, and scales much worse with
    the number of features/columns, but allows for faster incremental updates
    through 'partial_fit', which will give the same result as calls to fit.

    Note
    ----
    This model will not standardize the input data in any way.

    Note
    ----
    By default, will set the l1 and l2 regularization in the same way as
    GLMNet and SciKit-Learn - that is, the regularizations increase along
    with the number of rows in the data, which means they will be different
    after each call to 'fit' or 'partial_fit'. It is nevertheless possible
    to specify the l1 and l2 regularization directly, and both will remain
    constant that way, but be careful about the choice for such hyperparameters.

    Note
    ----
    Doing regression this way requires both memory and computation time
    which scale quadratically with the number of columns/features/variables. As
    such, the class will by default use C 'float' types (typically ``np.float32``)
    instead of C 'double' (``np.float64``), in order to save memory.

    Parameters
    ----------
    alpha : float
        Strenght of the regularization.
    l1_ratio : float [0,1]
        Proportion of the regularization that will be applied to the l1 norm of
        the coefficients (remainder will be applied to the l2 norm). Must be a
        number between zero and one. If passing ``l1_ratio=0``, it's recommended
        instead to use the ``LinearRegression`` class which uses more efficient
        procedures.

        Using higher l1 regularization is more likely to result in some of the
        obtained coefficients being exactly zero, which is oftentimes desirable.
    fit_intercept : bool
        Whether to add an intercept term to the formula. If passing 'True', it will
        be the last entry in the coefficients.
    l1 : None or float
        Strength of the l1 regularization. If passing it, will bypass the values
        set through ``alpha`` and ``l1_ratio``, and will remain constant inbetween
        calls to ``fit`` and ``partial_fit``. If passing this, should also pass
        ``l2`` or otherwise will assume that it is zero.
    l2 : None or float
        Strength of the l2 regularization. If passing it, will bypass the values
        set through ``alpha`` and ``l1_ratio``, and will remain constant inbetween
        calls to ``fit`` and ``partial_fit``. If passing this, should also pass
        ``l1`` or otherwise will assume that it is zero.
    use_float : bool
        Whether to use C 'float' type for the required matrices. If passing 'False',
        will use C 'double'. Be aware that memory usage for this model can grow
        very large. Can be changed after initialization.

    Attributes
    ----------
    coef_ : array(n) or array(n+1)
        The obtained coefficients. If passing 'fit_intercept=True', the intercept
        will be at the last entry.

    References
    ----------
    .. [1] Franc, Vojtech, Vaclav Hlavac, and Mirko Navara. "Sequential coordinate-wise algorithm for the non-negative least squares problem." International Conference on Computer Analysis of Images and Patterns. Springer, Berlin, Heidelberg, 2005.
    """
    def __init__(self, alpha=0.1, l1_ratio=0.5, fit_intercept=True,
                 l1=None, l2=None, use_float=True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.l1 = l1
        self.l2 = l2
        self._use_float = bool(use_float)

        self._set_dtype(force_cast=False)
        self.coef_ = np.empty(0, dtype=self._dtype)
        self._XtX = np.empty((0,0), dtype=self._dtype)
        self._XtY = np.empty(0, dtype=self._dtype)
        self._buffer_XtY2 = np.empty(0, dtype=self._dtype)
        self._prev_coef = np.empty(0, dtype=self._dtype)
        self._m = 0
        self._n = 0
        self._l1_curr = 0.
        self._l2_curr = 0.

        self.is_fitted_ = False

    def _set_dtype(self, force_cast=False):
        self._dtype = ctypes.c_float if self._use_float else ctypes.c_double

        if force_cast:
            if self.coef_.dtype != self._dtype:
                self.coef_ = self.coef_.astype(self._dtype)
            if self._XtX.dtype != self._dtype:
                self._XtX = self._XtX.astype(self._dtype)
            if self._XtY.dtype != self._dtype:
                self._XtY = self._XtY.astype(self._dtype)
            if self._buffer_XtY2.dtype != self._dtype:
                self._buffer_XtY2 = self._buffer_XtY2.astype(self._dtype)
            if self._prev_coef.dtype != self._dtype:
                self._prev_coef = self._prev_coef.astype(self._dtype)

    @property
    def use_float(self):
        return self._use_float
    @use_float.setter
    def use_float(self, value):
        self._use_float = use_float
        self._set_dtype(force_cast=True)

    def _validate_inputs(self):
        if isinstance(self.alpha, int):
            self.alpha = float(self.alpha)
        if isinstance(self.l1_ratio, int):
            self.l1_ratio = float(self.l1_ratio)
        assert self.alpha >= 0.
        assert self.l1_ratio >= 0.
        assert self.l1_ratio <= 1.

        self.fit_intercept = bool(self.fit_intercept)

        if self.l1 is not None:
            if isinstance(self.l1, int):
                self.l1 = float(self.l1)
            assert self.l1 >= 0.
            if self.l2 is None:
                self.l2 = 0.
        if self.l2 is not None:
            if isinstance(self.l2, int):
                self.l2 = float(self.l2)
            assert self.l2 >= 0.
            if self.l1 is None:
                self.l1 = 0.

    def _calc_l1_l2(self, m):
        if self.l1 is not None:
            return self.l1, self.l2
        div = 1. / m
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1. - self.l1_ratio)
        l1 /= div
        l2 /= div
        return l1, l2

    def fit(self, X, y, sample_weight=None):
        """
        Fit model to data

        Note
        ----
        Calling 'fit' will reset whatever previous data was there. For fitting
        the model incrementally to new data, use 'partial_fit' instead.

        Parameters
        ----------
        X : array(m,n) or CSR matrix(m, n)
            The covariates.
        y : array-like(m)
            The target variable.
        sample_weight : None or array-like(m)
            Observation weights for each row.

        Returns
        -------
        self

        """
        self._validate_inputs()
        X, Xcsr, y, w = LinearRegression._process_X_y_w(self, X, y, sample_weight)
        self._m = X.shape[0]
        self._n = X.shape[1]
        if min(self._m, self._n) == 0:
            raise ValueError("Cannot fit model to empty inputs.")
        l1, l2 = self._calc_l1_l2(self._m)

        n_plus_b = int(self._n + self.fit_intercept)
        self._XtX = np.zeros((n_plus_b, n_plus_b), dtype=self._dtype)
        self._XtY = np.zeros(n_plus_b, dtype=self._dtype)
        self._buffer_XtY2 = np.empty(2*n_plus_b, dtype=self._dtype)
        self._prev_coef = np.zeros(2*n_plus_b, dtype=self._dtype)
        self.coef_ = np.empty(n_plus_b, dtype=self._dtype)

        cy_funs = _wrapper_float if self._use_float else _wrapper_double
        cy_funs.update_matrices_noinv(
            X,
            y,
            w,
            self._XtX,
            self._XtY,
            Xcsr,
            bool(self.fit_intercept),
            True
        )
        if (Xcsr is None) and np.isfortran(X):
            self._XtX[:,:] = self._XtX.T
            if np.isfortran(self._XtX):
                self._XtX = np.ascontiguousarray(self._XtX)
        ### Do not regularize the intercept
        cy_funs.add_to_diag(self._XtX, self.fit_intercept, l2)
        cy_funs.fill_lower_triangle(self._XtX)
        self._buffer_XtY2[:n_plus_b] =  self._XtY - l1
        self._buffer_XtY2[n_plus_b:] = -self._XtY - l1
        ### Do not regularize the intercept
        if self.fit_intercept:
            self._buffer_XtY2[self._n] =  self._XtY[-1]
            self._buffer_XtY2[-1]      = -self._XtY[-1]
        cy_funs.solve_elasticnet(
            self._XtX,
            self._buffer_XtY2,
            self._prev_coef,
            self.coef_
        )

        self._l1_curr = l1
        self._l2_curr = l2
        self.is_fitted_ = True
        return self

    def partial_fit(self, X, y, sample_weight=None, *args, **kwargs):
        """
        Fit model incrementally to new data

        Parameters
        ----------
        X : array(m,n) or CSR matrix(m, n)
            The covariates.
        y : array-like(m)
            The target variable.
        sample_weight : None or array-like(m)
            Observation weights for each row.

        Returns
        -------
        self

        """
        if not self.is_fitted_:
            return self.fit(X, y, sample_weight)

        if X.shape[1] != self._n:
            raise ValueError("Number of columns in X doesn't match with previous.")
        m_add = X.shape[0]
        if m_add == 0:
            return self
        self._validate_inputs()
        X, Xcsr, y, w = LinearRegression._process_X_y_w(self, X, y, sample_weight)

        ### Note: the regularization changes at each step if imitating elasticnet
        l1, l2_target = self._calc_l1_l2(self._m + m_add)
        l2_add = l2_target - self._l2_curr
        n_plus_b = int(self._n + self.fit_intercept)

        cy_funs = _wrapper_float if self._use_float else _wrapper_double
        cy_funs.update_matrices_noinv(
            X,
            y,
            w,
            self._XtX,
            self._XtY,
            Xcsr,
            bool(self.fit_intercept),
            False
        )
        if (Xcsr is None) and np.isfortran(X):
            self._XtX[:,:] = self._XtX.T
            if np.isfortran(self._XtX):
                self._XtX = np.ascontiguousarray(self._XtX)
        if (np.abs(l2_add) > 1e-10):
            cy_funs.add_to_diag(self._XtX, self.fit_intercept, l2_add)
        cy_funs.fill_lower_triangle(self._XtX)
        self._buffer_XtY2[:n_plus_b] =  self._XtY - l1
        self._buffer_XtY2[n_plus_b:] = -self._XtY - l1
        ### Do not regularize the intercept
        if self.fit_intercept:
            self._buffer_XtY2[self._n] =  self._XtY[-1]
            self._buffer_XtY2[-1]      = -self._XtY[-1]
        ### TODO: for some reason, it gets a very bad result if the previous
        ### coefficients are reused, which should not be the case
        self._prev_coef[:] = 0.
        cy_funs.solve_elasticnet(
            self._XtX,
            self._buffer_XtY2,
            self._prev_coef,
            self.coef_
        )

        self._m += m_add
        self._l1_curr = l1
        self._l2_curr = l2_target
        return self

    def predict(self, X):
        """
        Make predictions on new data

        Parameters
        ----------
        X : array(m,n) or CSR matrix(m, n)
            The covariates.

        Returns
        -------
        y_hat : array(m)
            The predicted values given 'X'.
        """
        return LinearRegression.predict(self, X)
