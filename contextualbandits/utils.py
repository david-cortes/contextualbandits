import numpy as np, types, warnings, multiprocessing
from copy import deepcopy
from joblib import Parallel, delayed
import pandas as pd

_unexpected_err_msg = "Unexpected error. Please open an issue in GitHub describing what you were doing."

def _convert_decision_function_w_sigmoid(classifier):
    if 'decision_function' in dir(classifier):
        classifier.decision_function_w_sigmoid = types.MethodType(_decision_function_w_sigmoid, classifier)
        #### Note: the weird name is to avoid potential collisions with user-defined methods
    elif 'predict' in dir(classifier):
        classifier.decision_function_w_sigmoid = types.MethodType(_decision_function_w_sigmoid_from_predict, classifier)
    else:
        raise ValueError("Classifier must have at least one of 'predict_proba', 'decision_function', 'predict'.")
    return classifier

def _add_method_predict_robust(classifier):
    if 'predict_proba' in dir(classifier):
        classifier.predict_proba_robust = types.MethodType(_robust_predict_proba, classifier)
    if 'decision_function_w_sigmoid' in dir(classifier):
        classifier.decision_function_robust = types.MethodType(_robust_decision_function_w_sigmoid, classifier)
    elif 'decision_function' in dir(classifier):
        classifier.decision_function_robust = types.MethodType(_robust_decision_function, classifier)
    if 'predict' in dir(classifier):
        classifier.predict_robust = types.MethodType(_robust_predict, classifier)

    return classifier

def _robust_predict(self, X):
    try:
        return self.predict(X).reshape(-1)
    except:
        return np.zeros(X.shape[0])

def _robust_predict_proba(self, X):
    try:
        return self.predict_proba(X)
    except:
        return np.zeros((X.shape[0], 2))

def _robust_decision_function(self, X):
    try:
        return self.decision_function(X).reshape(-1)
    except:
        return np.zeros(X.shape[0])

def _robust_decision_function_w_sigmoid(self, X):
    try:
        return self.decision_function_w_sigmoid(X).reshape(-1)
    except:
        return np.zeros(X.shape[0])

def _decision_function_w_sigmoid(self, X):
    pred = self.decision_function(X).reshape(-1)
    _apply_sigmoid(pred)
    return pred

def _decision_function_w_sigmoid_from_predict(self, X):
    return self.predict(X).reshape(-1)

def _calculate_beta_prior(nchoices):
    return (3.0 / nchoices, 4)

def _check_bools(batch_train=False, assume_unique_reward=False):
    return bool(batch_train), bool(assume_unique_reward)

def _check_constructor_input(base_algorithm, nchoices, batch_train=False):
    if isinstance(base_algorithm, list):
        if len(base_algorithm) != nchoices:
            raise ValueError("Number of classifiers does not match with number of choices.")
        ### For speed reason, here it will not test if each classifier has the right methods
    else:
        assert ('fit' in dir(base_algorithm))
        assert ('predict_proba' in dir(base_algorithm)) or ('decision_function' in dir(base_algorithm)) or ('predict' in dir(base_algorithm))
        if batch_train:
            assert 'partial_fit' in dir(base_algorithm)

    assert nchoices >= 2
    assert isinstance(nchoices, int)

def _check_njobs(njobs):
    if njobs < 1:
        njobs = multiprocessing.cpu_count()
    if njobs is None:
        return 1
    assert isinstance(njobs, int)
    assert njobs >= 1
    return njobs


def _check_beta_prior(beta_prior, nchoices, default_b):
    if beta_prior == 'auto':
        out = (_calculate_beta_prior(nchoices), default_b)
    elif beta_prior is None:
        out = ((1.0,1.0), 0)
    else:
        assert len(beta_prior) == 2
        assert len(beta_prior[0]) == 2
        assert isinstance(beta_prior[1], int)
        assert isinstance(beta_prior[0][0], int) or isinstance(beta_prior[0][0], float)
        assert isinstance(beta_prior[0][1], int) or isinstance(beta_prior[0][1], float)
        assert (beta_prior[0][0] > 0) and (beta_prior[0][1] > 0)
        out = beta_prior
    return out

def _check_smoothing(smoothing):
    if smoothing is None:
        return None
    assert len(smoothing) >= 2
    assert (smoothing[0] >= 0) & (smoothing[1] >= 0)
    assert smoothing[1] > smoothing[0]
    return float(smoothing[0]), float(smoothing[1])


def _check_fit_input(X, a, r, choice_names = None):
    X = _check_X_input(X)
    a = _check_1d_inp(a)
    r = _check_1d_inp(r)
    assert X.shape[0] == a.shape[0]
    assert X.shape[0] == r.shape[0]
    if choice_names is not None:
        a = pd.Categorical(a, choice_names).codes
        if pd.isnull(a).sum() > 0:
            raise ValueError("Input contains actions/arms that this object does not have.")
    return X, a, r

def _check_X_input(X):
    if X.__class__.__name__ == 'DataFrame':
        X = X.values
    if type(X) == np.matrixlib.defmatrix.matrix:
        warnings.warn("'defmatrix' will be cast to array.")
        X = np.array(X)
    if type(X) != np.ndarray:
        raise ValueError("'X' must be a numpy array or pandas data frame.")
    if len(X.shape) == 1:
        X = X.reshape((1, -1))
    assert len(X.shape) == 2
    return X

def _check_1d_inp(y):
    if y.__class__.__name__ == 'DataFrame' or y.__class__.__name__ == 'Series':
        y = y.values
    if type(y) == np.matrixlib.defmatrix.matrix:
        warnings.warn("'defmatrix' will be cast to array.")
        y = np.array(y)
    if type(y) != np.ndarray:
        raise ValueError("'a' and 'r' must be numpy arrays or pandas data frames.")
    if len(y.shape) == 2:
        assert y.shape[1] == 1
        y = y.reshape(-1)
    assert len(y.shape) == 1
    return y

def _check_bay_inp(method, n_iter, n_samples):
    assert method in ['advi','nuts']
    if n_iter == 'auto':
        if method == 'nuts':
            n_iter = 100
        else:
            n_iter = 2000
    assert n_iter > 0
    if isinstance(n_iter, float):
        n_iter = int(n_iter)
    assert isinstance(n_iter, int)

    assert n_samples > 0
    if isinstance(n_samples, float):
        n_samples = int(n_samples)
    assert isinstance(n_samples, int)

    return n_iter, n_samples

def _check_active_inp(self, base_algorithm, f_grad_norm, case_one_class):
    if f_grad_norm == 'auto':
        _check_autograd_supported(base_algorithm)
        self._get_grad_norms = _get_logistic_grads_norms
    else:
        assert callable(f_grad_norm)
        self._get_grad_norms = f_grad_norm

    if case_one_class == 'auto':
        self._force_fit = False
        self._rand_grad_norms = _gen_random_grad_norms
    elif case_one_class == 'zero':
        self._force_fit = False
        self._rand_grad_norms = _gen_zero_norms
    elif case_one_class is None:
        self._force_fit = True
        self._rand_grad_norms = None
    else:
        assert callable(case_one_class)
        self._force_fit = False
        self._rand_grad_norms = case_one_class
    self.case_one_class = case_one_class

def _extract_regularization(base_algorithm):
    if base_algorithm.__class__.__name__ == 'LogisticRegression':
        return 1.0 / base_algorithm.C
    elif base_algorithm.__class__.__name__ == 'SGDClassifier':
        return base_algorithm.alpha
    elif base_algorithm.__class__.__name__ == 'RidgeClassifier':
        return base_algorithm.alpha
    elif base_algorithm.__class__.__name__ == 'StochasticLogisticRegression':
        return base_algorithm.reg_param
    else:
        raise ValueError("'auto' option only available for 'LogisticRegression', 'SGDClassifier', 'RidgeClassifier', and 'StochasticLogisticRegression' (this package's or stochQN's).")

def _logistic_grad_norm(X, y, pred, base_algorithm):
    coef = base_algorithm.coef_.reshape(-1)
    err = pred - y

    if X.__class__.__name__ in ['coo_matrix', 'csr_matrix', 'csc_matrix']:
        if X.__class__.__name__ != 'csr_matrix':
            from scipy.sparse import csr_matrix
            warnings.warn("Sparse matrix will be cast to CSR format.")
            X = csr_matrix(X)
        grad_norm = X.multiply(err)
    else:
        grad_norm = X * err.reshape((-1, 1))

    ### Note: since this is done on a row-by-row basis on two classes only,
    ### it doesn't matter whether the loss function is summed or averaged over
    ### data points, or whether there is regularization or not.

    ## coefficients
    grad_norm = np.linalg.norm(grad_norm, axis=1) ** 2

    ## intercept
    if base_algorithm.fit_intercept:
        grad_norm += err ** 2

    return grad_norm

def _get_logistic_grads_norms(base_algorithm, X, pred):
    return np.c_[_logistic_grad_norm(X, 0, pred, base_algorithm), _logistic_grad_norm(X, 1, pred, base_algorithm)]

def _check_autograd_supported(base_algorithm):
    assert base_algorithm.__class__.__name__ in ['LogisticRegression', 'SGDClassifier', 'RidgeClassifier', 'StochasticLogisticRegression']
    if base_algorithm.__class__.__name__ == 'LogisticRegression':
        if base_algorithm.penalty != 'l2':
            raise ValueError("Automatic gradients only defined for LogisticRegression with l2 regularization.")
        if base_algorithm.intercept_scaling != 1:
            raise ValueError("Automatic gradients for LogisticRegression not implemented with 'intercept_scaling'.")

    if base_algorithm.__class__.__name__ == 'RidgeClassifier':
        if base_algorithm.normalize:
            raise ValueError("Automatic gradients for LogisticRegression only implemented without 'normalize'.")

    if base_algorithm.__class__.__name__ == 'SGDClassifier':
        if base_algorithm.loss != 'log':
            raise ValueError("Automatic gradients for LogisticRegression only implemented with logistic loss.")
        if base_algorithm.penalty != 'l2':
            raise ValueError("Automatic gradients only defined for LogisticRegression with l2 regularization.")
    
    try:
        if base_algorithm.class_weight is not None:
            raise ValueError("Automatic gradients for LogisticRegression not supported with 'class_weight'.")
    except:
        pass

def _gen_random_grad_norms(X, n_pos, n_neg):
    ### Note: there isn't any theoretical reason behind these chosen distributions and numbers.
    ### A custom function might do a lot better.
    magic_number = np.log10(X.shape[1])
    smooth_prop = (n_pos + 1.0) / (n_pos + n_neg + 2.0)
    return np.c_[np.random.gamma(magic_number / smooth_prop, magic_number, size=X.shape[0]),
                 np.random.gamma(magic_number * smooth_prop, magic_number, size=X.shape[0])]

def _gen_zero_norms(X, n_pos, n_neg):
    return np.zeros((X.shape[0], 2))

def _apply_smoothing(preds, smoothing, counts):
    if (smoothing is not None) and (counts is not None):
        preds[:, :] = (preds * counts + smoothing[0]) / (counts + smoothing[1])
    return None

def _apply_sigmoid(x):
    if (len(x.shape) == 2):
        x[:, :] = 1.0 / (1.0 + np.exp(-x))
    else:
        x[:] = 1.0 / (1.0 + np.exp(-x))
    return None

def _apply_inverse_sigmoid(x):
    x[x == 0] = 1e-8
    x[x == 1] = 1 - 1e-8
    if (len(x.shape) == 2):
        x[:, :] = np.log(x / (1.0 - x))
    else:
        x[:] = np.log(x / (1.0 - x))
    return None

def _apply_softmax(x):
    x[:, :] = np.exp(x - x.max(axis=1).reshape((-1, 1)))
    x[:, :] = x / x.sum(axis=1).reshape((-1, 1))
    return None

class _FixedPredictor:
    def __init__(self):
        pass

    def fit(self, X=None, y=None, sample_weight=None):
        pass

    def decision_function_w_sigmoid(self, X):
        return self.decision_function(X)

class _BetaPredictor(_FixedPredictor):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def predict_proba(self, X):
        preds = np.random.beta(self.a, self.b, size = X.shape[0]).reshape((-1, 1))
        return np.c_[1.0 - preds, preds]

    def decision_function(self, X):
        return np.random.beta(self.a, self.b, size = X.shape[0])

    def predict(self, X):
        return (np.random.beta(self.a, self.b, size = X.shape[0])).astype('uint8')

    def predict_avg(self, X):
        pred = self.decision_function(X)
        _apply_inverse_sigmoid(pred)
        return pred

    def predict_rnd(self, X):
        return self.predict_avg(X)

    def predict_ucb(self, X):
        return self.predict_avg(X)

    def exploit(self, X):
        return np.repeat(self.a / self.b, X.shape[0])

class _ZeroPredictor(_FixedPredictor):

    def predict_proba(self, X):
        return np.c_[np.ones((X.shape[0], 1)),  np.zeros((X.shape[0], 1))]

    def decision_function(self, X):
        return np.zeros(X.shape[0])

    def predict(self, X):
        return np.zeros(X.shape[0])

    def predict_avg(self, X):
        return np.repeat(-1e6, X.shape[0])

    def predict_rnd(self, X):
        return self.predict_avg(X)

    def predict_ucb(self, X):
        return self.predict_avg(X)

class _OnePredictor(_FixedPredictor):

    def predict_proba(self, X):
        return np.c_[np.zeros((X.shape[0], 1)),  np.ones((X.shape[0], 1))]

    def decision_function(self, X):
        return np.ones(X.shape[0])

    def predict(self, X):
        return np.ones(X.shape[0])

    def predict_avg(self, X):
        return np.repeat(1e6, X.shape[0])

    def predict_rnd(self, X):
        return self.predict_avg(X)

    def predict_ucb(self, X):
        return self.predict_avg(X)

class _RandomPredictor(_FixedPredictor):
    def _gen_random(self, X):
        return np.random.random(size = X.shape[0])

    def predict(self, X):
        return (self._gen_random(X) >= .5).astype('uint8')

    def decision_function(self, X):
        return np.random.random(size = X.shape[0])

    def predict_proba(self, X):
        pred = self._gen_random(X)
        return np.c[pred, 1 - pred]

class _BootstrappedClassifierBase:
    def __init__(self, base, nsamples, percentile = 80, partialfit = False, partial_method = "gamma", njobs = 1):
        self.bs_algos = [deepcopy(base) for n in range(nsamples)]
        self.partialfit = partialfit
        self.partial_method = partial_method
        self.nsamples = nsamples
        self.percentile = percentile
        self.njobs = njobs

    def fit(self, X, y):
        ### Note: radom number generators are not always thread-safe, so don't parallelize this
        ix_take_all = np.random.randint(X.shape[0], size = (X.shape[0], self.nsamples))
        Parallel(n_jobs=self.njobs, verbose=0, require="sharedmem")(delayed(self._fit_single)(sample, ix_take_all, X, y) for sample in range(self.nsamples))

    def _fit_single(self, sample, ix_take_all, X, y):
        ix_take = ix_take_all[:, sample]
        xsample = X[ix_take, :]
        ysample = y[ix_take]
        nclass = ysample.sum()
        if not self.partialfit:
            if nclass == ysample.shape[0]:
                self.bs_algos[sample] = _OnePredictor()
                return None
            elif nclass == 0:
                self.bs_algos[sample] = _ZeroPredictor()
                return None
        self.bs_algos[sample].fit(xsample, ysample)

    def partial_fit(self, X, y, classes=None):
        if self.partial_method == "gamma":
            w_all = np.random.gamma(1, 1, size = (X.shape[0], self.nsamples))
            appear_times = None
            rng = None
        elif self.partial_method == "poisson":
            w_all = None
            appear_times = np.random.poisson(1, size = (X.shape[0], self.nsamples))
            rng = np.arange(X.shape[0])
        else:
            raise ValueError(_unexpected_err_msg)
        Parallel(n_jobs=self.njobs, verbose=0, require="sharedmem")(delayed(self._partial_fit_single)(sample, w_all, appear_times, rng, X, y) for sample in range(self.nsamples))

    def _partial_fit_single(self, sample, w_all, appear_times_all, rng, X, y):
        if w_all is not None:
            self.bs_algos[sample].partial_fit(X, y, classes=[0, 1], sample_weight=w_all[:, sample])
        elif appear_times_all is not None:
            appear_times = np.repeat(rng, appear_times_all[:, sample])
            xsample = X[appear_times]
            ysample = y[appear_times]
            self.bs_algos[sample].partial_fit(xsample, ysample, classes = [0, 1])
        else:
            raise ValueError(_unexpected_err_msg)

    def _score_max(self, X):
        pred = np.empty((X.shape[0], self.nsamples))
        Parallel(n_jobs=self.njobs, verbose=0, require="sharedmem")(delayed(self._assign_score)(sample, pred, X) for sample in range(self.nsamples))
        return np.percentile(pred, self.percentile, axis=1)

    def _score_avg(self, X):
        ### Note: don't try to make it more memory efficient by summing to a single array,
        ### as otherwise it won't be multithreaded.
        pred = np.empty((X.shape[0], self.nsamples))
        Parallel(n_jobs=self.njobs, verbose=0, require="sharedmem")(delayed(self._assign_score)(sample, pred, X) for sample in range(self.nsamples))
        return pred.mean(axis = 1)

    def _assign_score(self, sample, pred, X):
        pred[:, sample] = self._get_score(sample, X)

    def _score_rnd(self, X):
        chosen_sample = np.random.randint(self.nsamples)
        return self._get_score(chosen_sample, X)

    def exploit(self, X):
        return self._score_avg(X)

    def predict(self, X):
        ### Thompson sampling
        if self.percentile is None:
            pred = self._score_rnd(X)

        ### Upper confidence bound
        else:
            pred = self._score_max(X)

        return pred

class _BootstrappedClassifier_w_predict_proba(_BootstrappedClassifierBase):
    def _get_score(self, sample, X):
        return self.bs_algos[sample].predict_proba(X)[:, 1]

class _BootstrappedClassifier_w_decision_function(_BootstrappedClassifierBase):
    def _get_score(self, sample, X):
        pred = self.bs_algos[sample].decision_function(X).reshape(-1)
        _apply_sigmoid(pred)
        return pred

class _BootstrappedClassifier_w_predict(_BootstrappedClassifierBase):
    def _get_score(self, sample, X):
        return self.bs_algos[sample].predict(X).reshape(-1)

class _OneVsRest:
    def __init__(self, base, X, a, r, n, thr, alpha, beta, smooth=False, assume_un=False,
                 partialfit=False, force_fit=False, force_counters=False, njobs=1):
        if 'predict_proba' not in dir(base):
            base = _convert_decision_function_w_sigmoid(base)
        if partialfit:
            base = _add_method_predict_robust(base)
        if isinstance(base, list):
            self.base = None
            self.algos = base
        else:
            self.base = base
            self.algos = [deepcopy(base) for i in range(n)]
        self.n = n
        self.smooth = smooth
        self.assume_un = assume_un
        self.njobs = njobs
        self.force_fit = force_fit
        self.thr = thr
        self.partialfit = bool(partialfit)
        self.force_counters = bool(force_counters)
        if self.force_counters or (self.thr > 0 and not self.force_fit):
            ## in case it has beta prior, keeps track of the counters until no longer needed
            self.alpha = alpha
            self.beta = beta

            ## beta counters are represented as follows:
            # * first row: whether it shall use the prior
            # * second row: number of positives
            # * third row: number of negatives
            self.beta_counters = np.zeros((3, n))

        if self.smooth is not None:
            self.counters = np.zeros((1, n)) ##counters are row vectors to multiply them later with pred matrix
        else:
            self.counters = None

        if self.partialfit:
            self.partial_fit(X, a, r)
        else:
            Parallel(n_jobs=self.njobs, verbose=0, require="sharedmem")(delayed(self._full_fit_single)(choice, X, a, r) for choice in range(self.n))

    def _drop_arm(self, drop_ix):
        del self.algos[drop_ix]
        self.n -= 1
        if self.smooth is not None:
            self.counters = self.counters[:, np.arange(self.counters.shape[1]) != drop_ix]
        if self.force_counters or (self.thr > 0 and not self.force_fit):
            self.beta_counters = self.beta_counters[:, np.arange(self.beta_counters.shape[1]) != drop_ix]

    def _spawn_arm(self, fitted_classifier = None, n_w_req = 0, n_wo_rew = 0):
        self.n += 1
        if self.smooth is not None:
            self.counters = np.c_[self.counters, np.array([n_w_req + n_wo_rew]).reshape((1, 1)).astype(self.counters.dtype)]
        if self.force_counters or (self.thr > 0 and not self.force_fit):
            new_beta_col = np.array([0 if (n_w_req + n_wo_rew) < self.thr else 1, self.alpha + n_w_req, self.beta + n_wo_rew]).reshape((3, 1)).astype(self.beta_counters.dtype)
            self.beta_counters = np.c_[self.beta_counters, new_beta_col]
        if fitted_classifier is not None:
            if 'predict_proba' not in dir(fitted_classifier):
                fitted_classifier = _convert_decision_function_w_sigmoid(fitted_classifier)
            if partialfit:
                fitted_classifier = _add_method_predict_robust(fitted_classifier)
            self.algos.append(fitted_classifier)
        else:
            if self.force_fit or self.partialfit:
                if self.base is None:
                    raise ValueError("Must provide a classifier when initializing with different classifiers per arm.")
                self.algos.append( deepcopy(self.base) )
            else:
                if self.force_counters or (self.thr > 0 and not self.force_fit):
                    self.algos.append(_BetaPredictor(self.beta_counters[:, -1][1], self.beta_counters[:, -1][2]))
                else:
                    self.algos.append(_ZeroPredictor())

    def _update_beta_counters(self, yclass, choice):
        if (self.beta_counters[0, choice] == 0) or self.force_counters:
            n_pos = yclass.sum()
            self.beta_counters[1, choice] += n_pos
            self.beta_counters[2, choice] += yclass.shape[0] - n_pos
            if (self.beta_counters[1, choice] > self.thr) and (self.beta_counters[2, choice] > self.thr):
                self.beta_counters[0, choice] = 1

    def _full_fit_single(self, choice, X, a, r):
        yclass, this_choice = self._filter_arm_data(X, a, r, choice)
        n_pos = yclass.sum()
        if self.smooth is not None:
            self.counters[0, choice] += yclass.shape[0]
        if (n_pos < self.thr) or ((yclass.shape[0] - n_pos) < self.thr):
            if not self.force_fit:
                self.algos[choice] = _BetaPredictor(self.alpha + n_pos, self.beta + yclass.shape[0] - n_pos)
                return None
        if n_pos == 0:
            if not self.force_fit:
                self.algos[choice] = _ZeroPredictor()
                return None
        if n_pos == yclass.shape[0]:
            if not self.force_fit:
                self.algos[choice] = _OnePredictor()
                return None
        xclass = X[this_choice, :]
        self.algos[choice].fit(xclass, yclass)

        if self.force_counters or (self.thr > 0 and not self.force_fit):
            self._update_beta_counters(yclass, choice)


    def partial_fit(self, X, a, r):
        Parallel(n_jobs=self.njobs, verbose=0, require="sharedmem")(delayed(self._partial_fit_single)(choice, X, a, r) for choice in range(self.n))

    def _partial_fit_single(self, choice, X, a, r):
        yclass, this_choice = self._filter_arm_data(X, a, r, choice)
        if self.smooth is not None:
            self.counters[choice] += yclass.shape[0]

        xclass = X[this_choice, :]
        if (xclass.shape[0] > 0) or self.force_fit:
            self.algos[choice].partial_fit(xclass, yclass, classes = [0, 1])

        ## update the beta counters if needed
        if self.force_counters:
            self._update_beta_counters(yclass, choice)

    def _filter_arm_data(self, X, a, r, choice):
        if self.assume_un:
            this_choice = (a == choice)
            arms_w_rew = (r == 1)
            yclass = r[this_choice | arms_w_rew]
            yclass[arms_w_rew & (~this_choice) ] = 0
            this_choice = this_choice | arms_w_rew
        else:
            this_choice = (a == choice)
            yclass = r[this_choice]

        ## Note: don't filter X here as in many cases it won't end up used
        return yclass, this_choice

    def decision_function(self, X):
        preds = np.zeros((X.shape[0], self.n))
        Parallel(n_jobs=self.njobs, verbose=0, require="sharedmem")(delayed(self._decision_function_single)(choice, X, preds, 1) for choice in range(self.n))
        _apply_smoothing(preds, self.smooth, self.counters)
        return preds

    def _decision_function_single(self, choice, X, preds, depth=2):
        ## case when using partial_fit and need beta predictions
        if (self.partialfit or self.force_fit) and (self.thr > 0):
            if self.beta_counters[0, choice] == 0:
                preds[:, choice] = np.random.beta(self.alpha + self.beta_counters[1, choice],
                                                  self.beta  + self.beta_counters[2, choice],
                                                  size=preds.shape[0])
                return None

        if 'predict_proba_robust' in dir(self.algos[choice]):
            preds[:, choice] = self.algos[choice].predict_proba_robust(X)[:, 1]
        elif 'predict_proba' in dir(self.base):
            preds[:, choice] = self.algos[choice].predict_proba(X)[:, 1]
        else:
            if depth == 0:
                raise ValueError("This requires a classifier with method 'predict_proba'.")
            if 'decision_function_robust' in dir(self.algos[choice]):
                preds[:, choice] = self.algos[choice].decision_function_robust(X)
            elif 'decision_function_w_sigmoid' in dir(self.algos[choice]):
                preds[:, choice] = self.algos[choice].decision_function_w_sigmoid(X)
            else:
                preds[:, choice] = self.algos[choice].predict(X)

    def predict_proba(self, X):
        ### this is only used for softmax explorer
        preds = np.zeros((X.shape[0], self.n))
        Parallel(n_jobs=self.njobs, verbose=0, require="sharedmem")(delayed(self._decision_function_single)(choice, X, preds, 1) for choice in range(self.n))
        _apply_smoothing(preds, self.smooth, self.counters)
        _apply_inverse_sigmoid(preds)
        _apply_softmax(preds)
        return preds

    def predict_proba_raw(self,X):
        preds = np.zeros((X.shape[0], self.n))
        Parallel(n_jobs=self.njobs, verbose=0, require="sharedmem")(delayed(self._decision_function_single)(choice, X, preds, 0) for choice in range(self.n))
        _apply_smoothing(preds, self.smooth, self.counters)
        return preds

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)

    def should_calculate_grad(self, choice):
        if self.force_fit:
            return True
        if self.algos[choice].__class__.__name__ in ['_BetaPredictor', '_OnePredictor', '_ZeroPredictor']:
            return False
        if not bool(self.thr):
            return True
        try:
            return bool(self.beta_counters[0, choice])
        except:
            return True

    def get_n_pos(self, choice):
        return self.beta_counters[1, choice]

    def get_n_neg(self, choice):
        return self.beta_counters[2, choice]

    def exploit(self, X):
        ### only used with bootstrapped, bayesian, and lin-ucb/ts classifiers
        pred = np.empty((X.shape[0], self.n))
        Parallel(n_jobs=self.njobs, verbose=0, require="sharedmem")(delayed(self._exploit_single)(choice, pred, X) for choice in range(self.n))
        return pred

    def _exploit_single(self, choice, pred, X):
        pred[:, choice] = self.algos[choice].exploit(X)


class _BayesianLogisticRegression:
    def __init__(self, method='advi', niter=2000, nsamples=20, mode='ucb', perc=None):
        #TODO: reimplement with something faster than using PyMC3's black-box methods
        import pymc3 as pm, pandas as pd
        self.nsamples = nsamples
        self.niter = niter
        self.mode = mode
        self.perc = perc
        self.method = method

    def fit(self, X, y):
        with pm.Model():
            pm.glm.linear.GLM(X, y, family = 'binomial')
            pm.find_MAP()
            if self.method == 'advi':
                trace = pm.fit(progressbar = False, n = niter)
            if self.method == 'nuts':
                trace = pm.sample(progressbar = False, draws = niter)
        if self.method == 'advi':
            self.coefs = [i for i in trace.sample(nsamples)]
        elif self.method == 'nuts':
            samples_chosen = np.random.choice(np.arange( len(trace) ), size = nsamples, replace = False)
            samples_chosen = set(list(samples_chosen))
            self.coefs = [i for i in trace if i in samples_chosen]
        else:
            raise ValueError("'method' must be one of 'advi' or 'nuts'")
        self.coefs = pd.DataFrame.from_dict(coefs)
        self.coefs = coefs[ ['Intercept'] + ['x' + str(i) for i in range(X.shape[1])] ]
        self.intercept = coefs['Intercept'].values.reshape((-1, 1)).copy()
        del self.coefs['Intercept']
        self.coefs = coefs.values.T

    def _predict_all(self, X):
        pred_all = X.dot(self.coefs) + self.intercept
        _apply_sigmoid(pred_all)
        return pred_all

    def predict(self, X):
        pred = self._predict_all(X)
        if self.mode == 'ucb':
            pred = np.percentile(pred, self.perc, axis=1)
        elif self.mode == ' ts':
            pred = pred[:, np.random.randint(pred.shape[1])]
        else:
            raise ValueError(_unexpected_err_msg)
        return pred

    def exploit(self, X):
        pred = self._predict_all(X)
        return pred.mean(axis = 1)

class _LinUCBnTSSingle:
    def __init__(self, alpha, ts=False):
        self.alpha = alpha
        self.ts = ts

    def _sherman_morrison_update(self, Ainv, x):
        ## x should have shape (n, 1)
        Ainv -= np.linalg.multi_dot([Ainv, x, x.T, Ainv]) / (1.0 + np.linalg.multi_dot([x.T, Ainv, x]))

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        self.Ainv = np.eye(X.shape[1])
        self.b = np.zeros((X.shape[1], 1))

        self.partial_fit(X,y)

    def partial_fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        if 'Ainv' not in dir(self):
            self.Ainv = np.eye(X.shape[1])
            self.b = np.zeros((X.shape[1], 1))
        sumb = np.zeros((X.shape[1], 1))
        for i in range(X.shape[0]):
            x = X[i, :].reshape((-1, 1))
            r = y[i]
            sumb += r * x
            self._sherman_morrison_update(self.Ainv, x)

        self.b += sumb

    def predict(self, X, exploit=False):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        if self.ts:
            mu = (self.Ainv.dot(self.b)).reshape(-1)
            if not exploit:
                mu = np.random.multivariate_normal(mu, self.alpha * self.Ainv)
            return X.dot(mu).reshape(-1)

        else:
            pred = self.Ainv.dot(self.b).T.dot(X.T).reshape(-1)

            if not exploit:
                return pred

            for i in range(X.shape[0]):
                x = X[i, :].reshape((-1, 1))
                cb = self.alpha * np.sqrt(np.linalg.multi_dot([x.T, self.Ainv, x]))
                pred[i] += cb[0]

        return pred

    def exploit(self, X):
        return self.predict(X, exploit = True)
