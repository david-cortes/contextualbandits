import numpy as np
import ctypes
from scipy.sparse import issparse, isspmatrix_csr
import warnings
from sklearn.base import BaseEstimator
from . import _wrapper_double, _wrapper_float


class LinearRegression(BaseEstimator):
    """
    Linear Regression
    
    Typical Linear Regression model, which keeps track of the aggregated data
    needed to obtain the closed-form solution in a way that calling 'partial_fit'
    multiple times would be equivalent to a single call to 'fit' with all the data.

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
        is required for using the LinUCB and LinTS prediction modes. Ignored when
        passing ``method='sm'`` (the default). Note that is is possible to change
        the method after the object has already been fit.
    use_float : bool
        Whether to use C 'float' type for the required matrices. If passing 'False',
        will use C 'double'. Be aware that memory usage for this model can grow
        very large. Can be changed after initialization.
    copy_X : bool
        Whether to make deep copies of the 'X' input passed to the model's methods.
        If passing 'False', the 'X' object might be modified in-place. Note that
        if passing 'False', passing 'X' which is a non-contiguous subset of a
        larger array (e.g. 'X[:10, :100]') might provide the wrong results.

    Attributes
    ----------
    coef_ : array(n) or array(n+1)
        The obtained coefficients. If passing 'fit_intercept=True', the intercept
        will be at the last entry.
    """
    def __init__(self, lambda_=1., fit_intercept=True, method="sm", calc_inv=True,
                 use_float=True, copy_X=True):
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
        self._method = method
        self._calc_inv = bool(calc_inv)
        self._use_float = bool(use_float)
        self.copy_X = bool(copy_X)

        self._set_dtype(force_cast=False)
        self.coef_ = np.empty(0, dtype=self._dtype)
        self._XtX = np.empty((0,0), dtype=self._dtype)
        self._invXtX = np.empty((0,0), dtype=self._dtype)
        self._XtY = np.empty(0, dtype=self._dtype)
        self._bufferX = np.empty(0, dtype=self._dtype)
        self._n = 0

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
        if self._method != value:
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
        if bool(value) != self._calc_inv:
            if self._method == "chol":
                self._invXtX = np.empty(self._XtX.shape, dtype=self._dtype)
                cy_funs.get_matrix_inv(self._XtX, self._invXtX)
            self._calc_inv = bool(value)


    def _process_X_y_w(self, X, y, sample_weight, only_X=False):
        if X.dtype != self._dtype:
            X = X.astype(self._dtype)
        if not issparse(X):
            Xcsr = None
            if X.__class__.__name__ == "DataFrame":
                X = X.to_numpy()
            if self.copy_X:
                X = X.copy()
            X = np.array(X)
            if len(X.shape) != 2:
                raise ValueError("'X' must be 2-dimensional")
        else:
            if not isspmatrix_csr(X):
                warnings.warn("Sparse matrices only supported in CSR format. Input will be converted.")
                X = csr_matrix(X)
            Xcsr = X.copy()
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
            w = np.array(sample_weight).reshape(-1)

        if self.copy_X:
            y = y.copy()
            w = w.copy()

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
        else:
            self._invXtX, self._XtY, self.coef_ = \
                cy_funs.fit_model_inv(
                    X, y, w, Xcsr,
                    add_bias=self.fit_intercept,
                    lam = self.lambda_
                )

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
        alpha : float > 0
            The multiplier for the width of the bound.
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
        assert alpha > 0.
        if isinstance(alpha, int):
            alpha = float(alpha)
        assert isinstance(alpha, float)

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

        if (self.method == "chol") and (not self.calc_inv):
            raise ValueError("Not available when using 'method=\"chol\"' and 'calc_inv=False'.")

        tol = 1e-20
        if np.linalg.det(self._invXtX) >= tol:
            cov = self._invXtX
        else:
            cov = self._invXtX.copy()
            n = cov.shape[1]
            for i in range(10):
                cov[np.arange(n), np.arange(n)] += 1e-1
                if np.linalg.det(cov) >= tol:
                    break

        if sample_unique:
            if random_state is None:
                coef = np.random.multivariate_normal(mean=self.coef_,
                                                     cov=v_sq * cov,
                                                     size=X.shape[0])
            else:
                coef = random_state.multivariate_normal(mean=self.coef_,
                                                        cov=v_sq * cov,
                                                        size=X.shape[0])
            if not issparse(X):
                pred = np.einsum("ij,ij->i", X, coef[:,:X.shape[1]])
            else:
                pred = X.multiply(coef[:,:X.shape[1]]).sum(axis=1)
            if self.fit_intercept:
                pred[:] += coef[:,-1]
        else:
            if random_state is None:
                coef = np.random.multivariate_normal(mean=self.coef_,
                                                     cov=v_sq * cov)
            else:
                coef = random_state.multivariate_normal(mean=self.coef_,
                                                        cov=v_sq * cov)
            pred = X.dot(coef[:X.shape[1]])
            if self.fit_intercept:
                pred[:] += coef[-1]

        return pred
