import numpy as np, types
from copy import deepcopy

def _convert_decision_function(classifier):
    if 'decision_function' in dir(classifier):
        classifier.decision_function_original123 = deepcopy(classifier.decision_function)
        classifier.decision_function = types.MethodType(_converted_decision_function, classifier)
    return classifier

def _modify_predict_method(classifier):
    classifier.predict_old=deepcopy(classifier.predict)
    classifier.predict=types.MethodType(_robust_predict, classifier)

    if 'predict_proba' in dir(classifier):
        # TODO: once scikit-learn's issue 10938 is solved, make like the others
        classifier.predict_proba_new=types.MethodType(_robust_predict_proba, classifier)
    if 'decision_function' in dir(classifier):
        classifier.decision_function_old=classifier.decision_function
        classifier.decision_function=types.MethodType(_robust_decision_function, classifier)

    return classifier

def _robust_predict(self, X):
    if ('coef_' in dir(self)) or ('coefs_' in dir(self)):
        try:
            return self.predict_old(X)
        except:
            return np.zeros(X.shape[0])
    else:
        return np.zeros(X.shape[0])

def _robust_predict_proba(self, X):
    if ('coef_' in dir(self)) or ('coefs_' in dir(self)):
        try:
            return self.predict_proba(X)
        except:
            return np.zeros((X.shape[0],2))
    else:
        return np.zeros((X.shape[0],2))

def _robust_decision_function(self, X):
    if ('coef_' in dir(self)) or ('coefs_' in dir(self)):
        try:
            return self.decision_function_old(X)
        except:
            return np.zeros(X.shape[0])
    else:
        return np.zeros(X.shape[0])

def _converted_decision_function(self, X):
    pred = self.decision_function_original123(X)
    _apply_sigmoid(pred)
    return pred

def _calculate_beta_prior(nchoices):
    return (3/nchoices, 4)

def _check_bools(batch_train=False, assume_unique_reward=False):
    return bool(batch_train), bool(assume_unique_reward)

def _check_constructor_input(base_algorithm, nchoices, batch_train=False):
    assert nchoices > 2
    assert isinstance(nchoices, int)
    assert ('fit' in dir(base_algorithm)) and ('predict' in dir(base_algorithm))
    if batch_train:
        assert 'partial_fit' in dir(base_algorithm)
    return None

def _check_beta_prior(beta_prior, nchoices, default_b):
    if beta_prior=='auto':
        out = (_calculate_beta_prior(nchoices),default_b)
    elif beta_prior==None:
        out=((1,1),0)
    else:
        assert len(beta_prior)==2
        assert len(beta_prior[0])==2
        assert isinstance(beta_prior[1],int)
        assert isinstance(beta_prior[0][0],int) or isinstance(beta_prior[0][0],float)
        assert isinstance(beta_prior[0][1],int) or isinstance(beta_prior[0][1],float)
        assert (beta_prior[0][0]>0) and (beta_prior[0][1]>0)
        out = beta_prior
    return out

def _check_smoothing(smoothing):
    if smoothing is None:
        return None
    assert len(smoothing) >= 2
    assert (smoothing[0] >= 0) & (smoothing[1] >= 0)
    assert smoothing[1] > smoothing[0]
    return smoothing[0], smoothing[1]


def _check_fit_input(X,a,r):
    X=_check_X_input(X)
    a=_check_1d_inp(a)
    r=_check_1d_inp(r)
    assert X.shape[0]==a.shape[0]
    assert X.shape[0]==r.shape[0]
        
    return X,a,r

def _check_X_input(X):
    if type(X).__name__=='DataFrame':
        X=X.values
    if type(X)==np.matrixlib.defmatrix.matrix:
        X=np.array(X)
    if type(X)!=np.ndarray:
        raise ValueError("'X' must be a numpy array or pandas data frame.")
    if len(X.shape)==1:
        X=X.reshape(1,-1)
    assert len(X.shape)==2
    return X

def _check_1d_inp(y):
    if type(y).__name__=='DataFrame' or type(y).__name__=='Series':
        y=y.values
    if type(y)==np.matrixlib.defmatrix.matrix:
        y=np.array(y)
    if type(y)!=np.ndarray:
        raise ValueError("'a' and 'r' must be numpy arrays or pandas data frames.")
    if len(y.shape)==2:
        assert y.shape[1]==1
        y=y.reshape(-1)
    assert len(y.shape)==1
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
        return 1 / base_algorithm.C
    elif base_algorithm.__class__.__name__ == 'SGDClassifier':
        return base_algorithm.alpha
    elif base_algorithm.__class__.__name__ == 'RidgeClassifier':
        return base_algorithm.alpha
    else:
        raise ValueError("'auto' option only available for 'LogisticRegression', 'SGDClassifier', and 'RidgeClassifier'.")
    return None

def _logistic_grad_norm(X, y, pred, base_algorithm):
    coef = base_algorithm.coef_.reshape(-1)
    intercept = base_algorithm.intercept_
    err = pred - y

    if X.__class__.__name__ in ['coo_matrix', 'csr_matrix', 'csc_matrix']:
        if X.__class__.__name__ != 'csr_matrix':
            from scipy.sparse import csr_matrix
            X = csr_matrix(X)
        grad_norm = X.multiply(err)
        is_sp = True
    else:
        grad_norm = X * err.reshape((-1, 1))
        is_sp = False

    ## coefficients
    grad_norm = np.linalg.norm(grad_norm, axis=1) ** 2

    ## intercept
    grad_norm += err ** 2

    return grad_norm

def _get_logistic_grads_norms(base_algorithm, X, pred):
    return np.c_[_logistic_grad_norm(X, 0, pred, base_algorithm), _logistic_grad_norm(X, 1, pred, base_algorithm)]

def _check_autograd_supported(base_algorithm):
    assert base_algorithm.__class__.__name__ in ['LogisticRegression', 'SGDClassifier', 'RidgeClassifier']
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

    if not base_algorithm.fit_intercept:
        raise ValueError("Automatic gradients only defined for LogisticRegression with intercept.")
    if base_algorithm.class_weight is not None:
        raise ValueError("Automatic gradients for LogisticRegression not supported with 'class_weight'.")

def _gen_random_grad_norms(X, n_pos, n_neg):
    magic_number = np.log10(X.shape[1])
    smooth_prop = (n_pos + 1) / (n_pos + n_neg + 2)
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
        x[:, :] = 1 / (1 + np.exp(-x))
    else:
        x[:] = 1 / (1 + np.exp(-x))
    return None

def _apply_inverse_sigmoid(x):
    x[x == 0] = 1e-8
    x[x == 1] = 1 - 1e-8
    if (len(x.shape) == 2):
        x[:, :] = np.log(x / (1 - x))
    else:
        x[:] = np.log(x / (1 - x))
    return None

def _apply_softmax(x):
    x[:, :] = np.exp(x - x.max(axis=1).reshape((-1, 1)))
    x[:, :] = x / x.sum(axis=1).reshape((-1, 1))
    return None

def _update_beta_counters(beta_counters, yclass, choice, thr, force_counters=False):
    if (beta_counters[0, choice] == 0) or force_counters:
        n_pos = yclass.sum()
        beta_counters[1, choice] += n_pos
        beta_counters[2, choice] += yclass.shape[0] - n_pos
        if (beta_counters[1, choice] > thr) and (beta_counters[2, choice] > thr):
            beta_counters[0, choice] = 1

class _BetaPredictor:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def fit(self, X=None, y=None, sample_weight=None):
        pass
    
    def predict_proba(self, X):
        preds = np.random.beta(self.a, self.b, size=X.shape[0]).reshape(-1,1)
        return np.c_[1 - preds, preds]
    
    def decision_function(self, X):
        return np.random.beta(self.a, self.b, size=X.shape[0])
    
    def predict(self, X):
        return (np.random.beta(self.a, self.b, size=X.shape[0])).astype('uint8')
    
    def predict_avg(self, X):
        pred = self.decision_function(X)
        _apply_inverse_sigmoid(pred)
        return pred
    
    def predict_rnd(self,X):
        return self.predict_avg(X)
    
    def predict_ucb(self,X):
        return self.predict_avg(X)
    
class _ZeroPredictor:
    def __init__(self):
        pass
        
    def fit(self,X=None,y=None, sample_weight=None):
        pass
    
    def predict_proba(self,X):
        return np.c_[np.ones((X.shape[0],1)),np.zeros((X.shape[0],1))]
    
    def decision_function(self,X):
        return np.zeros(X.shape[0])
    
    def predict(self,X):
        return np.zeros(X.shape[0])
    
    def predict_avg(self, X):
        return np.repeat(-1e10, X.shape[0])
    
    def predict_rnd(self, X):
        return self.predict_avg(X)
    
    def predict_ucb(self, X):
        return self.predict_avg(X)
    
class _OnePredictor:
    def __init__(self):
        pass
        
    def fit(self, X=None, y=None, sample_weight=None):
        pass
    
    def predict_proba(self, X):
        return np.c_[np.zeros((X.shape[0],1)),np.ones((X.shape[0],1))]
    
    def decision_function(self, X):
        return np.ones(X.shape[0])
    
    def predict(self, X):
        return np.ones(X.shape[0])
    
    def predict_avg(self, X):
        return np.repeat(1e10, X.shape[0])
    
    def predict_rnd(self, X):
        return self.predict_avg(X)
    
    def predict_ucb(self, X):
        return self.predict_avg(X)
    
class _RandomPredictor:
    def __init__(self):
        pass
    
    def fit(self, X=None, y=None, sample_weight=None):
        pass
    
    def predict(self, X):
        return (np.random.random(size=X.shape[0]) >= .5).astype('uint8')
    
    def predict_proba(self, X):
        return np.random.random(size=X.shape[0])
    
    def predict_avg(self, X):
        pred = self.predict_proba(X)
        _apply_inverse_sigmoid(pred)
        return pred

    def predict_rnd(self, X):
        return self.predict_avg(X)
    
    def predict_ucb(self, X):
        return self.predict_avg(X)

class _ArrBSClassif:
    #TODO: refactor this out, make two classes, one for UCB and one for TS, which would then be embedded into _OneVsRest.
    def __init__(self,base,X,a,r,n,thr,alpha,beta,samples,smooth=False,assume_un=False,
                 partialfit=False,partial_method='gamma'):
        if partialfit:
            base = _modify_predict_method(base)
        if 'predict_proba' not in dir(base):
            base = _convert_decision_function(base)

        self.base=base
        self.algos=[[deepcopy(base) for i in range(samples)] for j in range(n)]
        self.n=n
        self.samples=samples
        self.partial_method=partial_method
        self.smooth=smooth
        self.assume_un=assume_un
        self.thr = thr
        self.partialfit = bool(partialfit)
        if self.smooth:
            self.counters = np.zeros((1,n)) ##counters are row vectors to multiply them later with pred matrix
        else:
            self.counters = None

        if self.partialfit:
            ## in case it has beta prior, keeps track of the counters until no longer needed
            self.alpha = alpha
            self.beta = beta
            if self.thr > 0:
                # first row: whether it shall use the prior
                # second row: number of positives
                # third row: number of negatives
                self.beta_counters = np.zeros((3, n))
            self.partial_fit(X, a, r)
        else:
            for choice in range(n):
                if self.assume_un:
                    this_choice=(a==choice)
                    arms_w_rew=(r==1)
                    yclass=r[this_choice|arms_w_rew]
                    yclass[arms_w_rew&(~this_choice)]=0
                    this_choice=this_choice|arms_w_rew
                else:
                    this_choice=(a==choice)
                    yclass=r[this_choice]
                if self.smooth is not None:
                    self.counters[0, choice] += yclass.shape[0]

                n_pos = yclass.sum()
                if (n_pos < thr) or ((yclass.shape[0] - n_pos) < thr):
                    self.algos[choice] = _BetaPredictor(alpha + n_pos, beta + yclass.shape[0] - n_pos)
                    continue
                if n_pos == 0:
                    self.algos[choice] =_ZeroPredictor()
                    continue
                if n_pos == yclass.shape[0]:
                    self.algos[choice] = _OnePredictor()
                    continue

                xclass = X[this_choice, :]
                for sample in range(samples):
                    ix_take = np.random.randint(xclass.shape[0], size=xclass.shape[0])
                    xsample = xclass[ix_take, :]
                    ysample = yclass[ix_take]
                    nclass = ysample.sum()
                    if (nclass == ysample.shape[0]) : and not self.partialfit
                        self.algos[choice][sample] = _OnePredictor()
                    elif (ysample.sum() > 0) or self.partialfit:
                        self.algos[choice][sample].fit(xsample, ysample)
                    else:
                        self.algos[choice][sample] = _ZeroPredictor()
                    
    def partial_fit(self,X,a,r):
        for choice in range(self.n):
            if self.assume_un:
                this_choice=(a==choice)
                arms_w_rew=(r==1)
                yclass=r[this_choice|arms_w_rew]
                yclass[arms_w_rew&(~this_choice)]=0
                this_choice=this_choice|arms_w_rew
            else:
                this_choice=(a==choice)
                yclass=r[this_choice]

            if self.smooth is not None:
                self.counters[0, choice] += yclass.shape[0]

            xclass = X[this_choice, :]
            if xclass.shape[0] > 0:
                for sample in range(self.samples):
                    if self.partial_method == 'poisson':
                        appear_times = np.repeat(np.arange(xclass.shape[0]), np.random.poisson(1, size = xclass.shape[0]))
                        xsample = xclass[appear_times]
                        ysample = yclass[appear_times]
                        self.algos[choice][sample].partial_fit(xsample, ysample, classes=[0, 1])
                    else:
                        self.algos[choice][sample].partial_fit(xclass, yclass,
                                classes=[0,1], sample_weight=np.random.gamma(1, 1, size = xclass.shape[0]))

                ## update the beta counters if needed
                if self.thr > 0:
                    _update_beta_counters(self.beta_counters, yclass, choice, self.thr)
    
    def score_avg(self,X):
        preds=np.zeros((X.shape[0], self.n))
        for choice in range(self.n):

            ## case when no model has been fit, uses dummy predictors from module
            if not (type(self.algos[choice]) == list):
                preds[:, choice] = self.algos[choice].decision_function(X)
                continue

            ## case when using partial_fit and need beta predictions
            if self.partialfit:
                if (self.thr > 0) and (self.beta_counters[0, choice] == 0):
                    preds[:, choice] = np.random.beta(self.alpha + self.beta_counters[1, choice],
                                                      self.beta + self.beta_counters[2, choice],
                                                      size = preds.shape[0])
                    continue

            ## case when there are fitted models
            for sample in range(self.samples):
                if 'predict_proba_new' in dir(self.algos[choice][sample]):
                    preds[:, choice] += self.algos[choice][sample].predict_proba_new(X)[:, 1]
                elif 'predict_proba' in dir(self.algos[choice][sample]):
                    preds[:, choice] += self.algos[choice][sample].predict_proba(X)[:, 1]
                elif 'decision_function' in dir(self.algos[choice][sample]):
                    preds[:, choice] += self.algos[choice][sample].decision_function(X)
                else:
                    preds[:, choice] += self.algos[choice][sample].predict(X)
                
            preds[:,choice] = preds[:,choice] / self.samples
        
        _apply_smoothing(preds, self.smooth, self.counters)
        return preds
    
    def score_max(self, X, perc=100):
        preds = np.zeros( (X.shape[0], self.n) )
        for choice in range(self.n):
            ## case when no model has been fit, uses dummy predictors from module
            if not (type(self.algos[choice]) == list):
                preds[:, choice] = self.algos[choice].decision_function(X)
                continue

            ## case when using partial_fit and need beta predictions
            if self.partialfit:
                if (self.thr > 0) and (self.beta_counters[0, choice] == 0):
                    preds[:, choice] = np.random.beta(self.alpha + self.beta_counters[1, choice],
                                                      self.beta + self.beta_counters[2, choice],
                                                      size=preds.shape[0])
                    continue

            ## case when there are fitted models
            arr_compare = list()
            for sample in range(self.samples):
                if 'predict_proba_new' in dir(self.algos[choice][sample]):
                    arr_compare.append(self.algos[choice][sample].predict_proba_new(X)[:, 1])
                elif 'predict_proba' in dir(self.algos[choice][sample]):
                    arr_compare.append(self.algos[choice][sample].predict_proba(X)[:, 1])
                elif 'decision_function' in dir(self.algos[choice][sample]):
                    arr_compare.append(self.algos[choice][sample].decision_function(X))
                else:
                    arr_compare.append(self.algos[choice][sample].predict(X))
                
            if perc == 100:
                preds[:, choice] = np.vstack(arr_compare).max(axis=0)
            else:
                preds[:, choice] = np.percentile(np.vstack(arr_compare), perc, axis=0)
        
        _apply_smoothing(preds, self.smooth, self.counters)
        return preds
    
    def score_rnd(self, X):
        preds = np.zeros( (X.shape[0], self.n) )
        for choice in range(self.n):
            ## case when no model has been fit, uses dummy predictors from module
            if type(self.algos[choice]) != list:
                preds[:, choice] = self.algos[choice].decision_function(X)
                continue
            ## case when using partial_fit and need beta predictions
            if self.partialfit:
                if (self.thr > 0) and (self.beta_counters[0, choice] == 0):
                    preds[:, choice] = np.random.beta(self.alpha + self.beta_counters[1, choice],
                                                      self.beta + self.beta_counters[2, choice],
                                                      size=preds.shape[0])
                    continue
            ## case when there are fitted models
            sample_take = np.random.randint(self.samples)
            if 'predict_proba_new' in dir(self.algos[choice][sample_take]):
                preds[:, choice] = self.algos[choice][sample_take].predict_proba_new(X)[:, 1]
            elif 'predict_proba' in dir(self.algos[choice][sample_take]):
                preds[:, choice] = self.algos[choice][sample_take].predict_proba(X)[:, 1]
            elif 'decision_function' in dir(self.algos[choice][sample_take]):
                preds[:, choice] = self.algos[choice][sample_take].decision_function(X)
            else:
                preds[:, choice] = self.algos[choice][sample_take].predict(X)
        
        _apply_smoothing(preds, self.smooth, self.counters)
        return preds
    
class _OneVsRest:
    def __init__(self, base, X, a, r, n, thr, alpha, beta, smooth=False, assume_un=False,
                 partialfit=False, force_fit=False, force_counters=False):
        if partialfit:
            base = _modify_predict_method(base)
        if 'predict_proba' not in dir(base):
            base = _convert_decision_function(base)
        self.base = base
        self.algos = [deepcopy(base) for i in range(n)]
        self.n = n
        self.smooth = smooth
        self.assume_un = assume_un
        self.force_fit = force_fit
        self.thr = thr
        self.partialfit = bool(partialfit)
        self.force_counters = bool(force_counters)
        if ((force_fit or partialfit) and (thr > 0)) or force_counters:
            ## in case it has beta prior, keeps track of the counters until no longer needed
            self.alpha = alpha
            self.beta = beta
            # first row: whether it shall use the prior
            # second row: number of positives
            # third row: number of negatives
            self.beta_counters = np.zeros((3, n))

        if self.smooth is not None:
            self.counters = np.zeros((1, n)) ##counters are row vectors to multiply them later with pred matrix
        else:
            self.counters = None
            
        if self.partialfit:
            self.partial_fit(X, a, r)
        else:
            for choice in range(n):
                if self.assume_un:
                    this_choice=(a==choice)
                    arms_w_rew=(r==1)
                    yclass=r[this_choice|arms_w_rew]
                    yclass[arms_w_rew&(~this_choice)]=0
                    this_choice=this_choice|arms_w_rew
                else:
                    this_choice = (a == choice)
                    yclass = r[this_choice]

                n_pos = yclass.sum()
                if self.smooth is not None:
                    self.counters[0, choice] += yclass.shape[0]
                if (n_pos < thr) or ((yclass.shape[0] - n_pos) < thr):
                    if not force_fit:
                        self.algos[choice] = _BetaPredictor(alpha + n_pos, beta + yclass.shape[0] - n_pos)
                        continue
                if n_pos == 0:
                    if not force_fit:
                        self.algos[choice] = _ZeroPredictor()
                        continue
                if n_pos == yclass.shape[0]:
                    if not force_fit:
                        self.algos[choice] = _OnePredictor()
                        continue
                xclass = X[this_choice, :]
                self.algos[choice].fit(xclass, yclass)

                if (self.force_fit and (thr > 0)) or (self.force_counters):
                    _update_beta_counters(self.beta_counters, yclass, choice, thr, self.force_counters)

                
    def partial_fit(self, X, a, r):
        for choice in range(self.n):
            if self.assume_un:
                this_choice=(a==choice)
                arms_w_rew=(r==1)
                yclass=r[this_choice|arms_w_rew]
                yclass[arms_w_rew&(~this_choice)]=0
                this_choice=this_choice|arms_w_rew
            else:
                this_choice=(a==choice)
                yclass=r[this_choice]

            if self.smooth is not None:
                self.counters[choice] += yclass.shape[0]

            xclass = X[this_choice,:]
            if (xclass.shape[0] > 0) or self.force_fit:
                self.algos[choice].partial_fit(xclass, yclass, classes = [0, 1])

            ## update the beta counters if needed
            if (self.thr > 0) or self.force_counters:
                _update_beta_counters(self.beta_counters, yclass, choice, self.thr, self.force_counters)
    
    def decision_function(self, X):
        preds = np.zeros((X.shape[0], self.n))
        for choice in range(self.n):

            ## case when using partial_fit and need beta predictions
            if (self.partialfit or self.force_fit) and (self.thr > 0):
                if self.beta_counters[0, choice] == 0:
                    preds[:, choice] = np.random.beta(self.alpha + self.beta_counters[1, choice],
                                                      self.beta + self.beta_counters[2, choice],
                                                      size=preds.shape[0])
                    continue

            if 'predict_proba_new' in dir(self.base):
                preds[:, choice] = self.algos[choice].predict_proba_new(X)[:, 1]
            elif 'predict_proba' in dir(self.base):
                preds[:, choice] = self.algos[choice].predict_proba(X)[:, 1]
            elif 'decision_function' in dir(self.base):
                preds[:, choice] = self.algos[choice].decision_function(X)
            else:
                preds[:, choice] = self.algos[choice].predict(X)

        _apply_smoothing(preds, self.smooth, self.counters)
        return preds
    
    def predict_proba(self, X):
        preds = np.zeros((X.shape[0],self.n))
        for choice in range(self.n):
            ## case when using partial_fit and need beta predictions
            if (self.partialfit or self.force_fit) and (self.thr > 0):
                if self.beta_counters[0, choice] == 0:
                    preds[:, choice] = np.random.beta(self.alpha + self.beta_counters[1, choice],
                                                      self.beta + self.beta_counters[2, choice],
                                                      size = preds.shape[0])
                    continue

            if 'predict_proba_new' in dir(self.base):
                preds[:, choice] = self.algos[choice].predict_proba_new(X)[:, 1]
            elif 'predict_proba' in dir(self.base):
                preds[:, choice] = self.algos[choice].predict_proba(X)[:, 1]
            elif 'decision_function' in dir(self.base):
                preds[:, choice] = self.algos[choice].decision_function(X)
            else:
                preds[:, choice] = self.algos[choice].predict(X)
        
        _apply_smoothing(preds, self.smooth, self.counters)
        _apply_inverse_sigmoid(preds)
        _apply_softmax(preds)
        return preds
    
    def predict_proba_raw(self,X):
        preds = np.zeros((X.shape[0],self.n))
        for choice in range(self.n):
            ## case when using partial_fit and need beta predictions
            if (self.partialfit or self.force_fit) and (self.thr > 0):
                if self.beta_counters[0, choice] == 0:
                    preds[:, choice] = np.random.beta(self.alpha + self.beta_counters[1, choice],
                                                      self.beta + self.beta_counters[2, choice],
                                                      size = preds.shape[0])
                    continue
            if 'predict_proba_new' in dir(self.algos[choice]):
                preds[:,choice]=self.algos[choice].predict_proba_new(X)[:, 1]
            elif 'predict_proba' in dir(self.algos[choice]):
                preds[:,choice]=self.algos[choice].predict_proba(X)[:, 1]
            else:
                raise ValueError("This requires a classifier with method 'predict_proba'.")
                
        _apply_smoothing(preds, self.smooth, self.counters)
        return preds
    
    def predict(self,X):
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
        X = _check_X_input(X)
        y = _check_1d_inp(y)
        assert X.shape[0] == y.shape[0]

        with pm.Model():
            pm.glm.linear.GLM(X, y, family='binomial')
            pm.find_MAP()
            if self.method == 'advi':
                trace = pm.fit(progressbar = False, n=niter)
            if self.method == 'nuts':
                trace = pm.sample(progressbar = False, draws=niter)
        if self.method == 'advi':
            self.coefs = [i for i in trace.sample(nsamples)]
        elif self.method == 'nuts':
            samples_chosen = np.random.choice(np.arange(len(trace)), size=nsamples, replace=False)
            samples_chosen = set(list(samples_chosen))
            self.coefs = [i for i in trace if i in samples_chosen]
        else:
            raise ValueError("'method' must be one of 'advi' or 'nuts'")
        self.coefs = pd.DataFrame.from_dict(coefs)
        self.coefs = coefs[ ['Intercept'] + ['x'+str(i) for i in range(X.shape[1])] ]
        self.intercept = coefs['Intercept'].values.reshape((-1,1)).copy()
        del self.coefs['Intercept']
        self.coefs = coefs.values.T
        
    def _predict_all(self, X):
        pred_all = X.dot(self.coefs) + self.intercept
        _apply_sigmoid(pred_all)
        return pred_all

    def predict_proba(self, X):
        pred = self._predict_all(X)
        if self.mode == 'ucb':
            pred = np.percentile(pred, self.perc, axis=1)
        elif self.mode == ' ts':
            pred = pred[:, np.random.randint(pred.shape[1])]
        else:
            pred = pred.mean(axis=1)
        
        return np.c_[1 - pred, pred]

    def predict_proba(self, X):
        pred = self._predict_proba()

class _LinUCBSingle:
    def __init__(self, alpha):
        self.alpha=alpha
        
    def fit(self, X, y):
        if len(X.shape)==1:
            X=X.reshape(1,-1)
        self.Ainv=np.eye(X.shape[1])
        self.b=np.zeros((X.shape[1], 1))
        
        self.partial_fit(X,y)
    
    def partial_fit(self, X, y):
        if len(X.shape)==1:
            X=X.reshape(1,-1)
        if 'Ainv' not in dir(self):
            self.Ainv=np.eye(X.shape[1])
            self.b=np.zeros((X.shape[1], 1))
        sumb=np.zeros((X.shape[1], 1))
        for i in range(X.shape[0]):
            x=X[i,:].reshape(-1,1)
            r=y[i]
            sumb+=r*x
            
            self.Ainv -= np.linalg.multi_dot([self.Ainv, x, x.T, self.Ainv])/\
                            (1 + np.linalg.multi_dot([x.T, self.Ainv, x]))
        
        self.b+=sumb
    
    def predict(self, X, exploit=False):
        if len(X.shape)==1:
            X=X.reshape(1,-1)
        
        pred=self.Ainv.dot(self.b).T.dot(X.T).reshape(-1)
        
        if not exploit:
            return pred
        
        for i in range(X.shape[0]):
            x=X[i,:].reshape(-1,1)
            cb=self.alpha*np.sqrt(np.linalg.multi_dot([x.T, self.Ainv, x]))
            pred[i]+=cb[0]
        
        return pred
