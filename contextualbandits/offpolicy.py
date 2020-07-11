# -*- coding: utf-8 -*-

import numpy as np
from .utils import _check_constructor_input, _check_fit_input, \
    _check_X_input, _check_1d_inp, _check_beta_prior, _check_smoothing, \
    _check_njobs, _check_random_state, _OnePredictor, _ZeroPredictor, _RandomPredictor
from .online import SeparateClassifiers
from copy import deepcopy
from joblib import Parallel, delayed

__all__ = ["DoublyRobustEstimator", "OffsetTree"]

class DoublyRobustEstimator:
    """
    Doubly-Robust Estimator
    
    Estimates the expected reward for each arm, applies a correction for the actions that
    were chosen, and converts the problem to const-sensitive classification, on which the
    base algorithm is then fit.
    
    Note
    ----
    This technique converts the problem into a cost-sensitive classification problem
    by calculating a matrix of expected rewards and turning it into costs. The base
    algorithm is then fit to this data, using either the Weighted All-Pairs approach,
    which requires a binary classifier with sample weights as base algorithm, or the
    Regression One-Vs-Rest approach, which requires a regressor as base algorithm.
    
    In the Weighted All-Pairs approach, this technique will fail if there are actions that
    were never taken by the exploration policy, as it cannot construct a model for them.
    
    The expected rewards are estimated with the imputer algorithm passed here, which should
    output a number in the range [0,1].
    
    This technique is meant for the case of contiunous rewards in the [0,1] interval,
    but here it is used for the case of discrete rewards {0,1}, under which it performs
    poorly. It is not recommended to use, but provided for comparison purposes.
    
    
    Alo important: this method requires to form reward estimates of all arms for each observation. In order to
    do so, you can either provide estimates as an array (see Parameters), or pass a model.
    
    One method to obtain reward estimates is to fit a model to the data and use its predictions as
    reward estimates. You can do so by passing an object of class
    `contextualbandits.online.SeparateClassifiers` which should be already fitted, or by passing a
    classifier with a 'predict_proba' method, which will be put into a 'SeparateClassifiers'
     object and fit to the same data passed to this function to obtain reward estimates.
    
    The estimates can make invalid predictions if there are some arms for which every time
    they were chosen they resulted in a reward, or never resulted in a reward. In such cases,
    this function includes the option to impute the "predictions" for them (which would otherwise
    always be exactly zero or one regardless of the context) by replacing them with random
    numbers ~Beta(3,1) or ~Beta(1,3) for the cases of always good and always bad.
    
    This is just a wild idea though, and doesn't guarantee reasonable results in such siutation.
    
    Note that, if you are using the 'SeparateClassifiers' class from the online module in this
    same package, it comes with a method 'predict_proba_separate' that can be used to get reward
    estimates. It still can suffer from the same problem of always-one and always-zero predictions though.
    
    Parameters
    ----------
    base_algorithm : obj
        Base algorithm to be used for cost-sensitive classification.
    reward_estimator : obj or array (n_samples, n_choices)
        One of the following:
            * An array with the first column corresponding to the reward estimates for the action chosen
              by the new policy, and the second column corresponding to the reward estimates for the
              action chosen in the data (see Note for details).
            * An already-fit object of class 'contextualbandits.online.SeparateClassifiers', which will
              be used to make predictions on the actions chosen and the actions that the new
              policy would choose.
            * A classifier with a 'predict_proba' method, which will be fit to the same test data
              passed here in order to obtain reward estimates (see Note 2 for details).
    nchoices : int
        Number of arms/labels to choose from.
        Only used when passing a classifier object to 'reward_estimator'.
    method : str, either 'rovr' or 'wap'
        Whether to use Regression One-Vs-Rest or Weighted All-Pairs (see Note 1)
    handle_invalid : bool
        Whether to replace 0/1 estimated rewards with randomly-generated numbers (see Note 2)
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        This is used when passing ``handle_invalid=True`` or ``beta_prior != None``.
    c : None or float
        Constant by which to multiply all scores from the exploration policy.
    pmin : None or float
        Scores (from the exploration policy) will be converted to the minimum between
        pmin and the original estimate.
    beta_prior : tuple((a, b), n), str "auto", or None
        Beta prior to pass to 'SeparateClassifiers'. Only used when passing to
        'reward_estimator' a classifier with 'predict_proba'. See the documentation
        of 'SeparateClassifiers' for details about it.
    smoothing : tuple(a, b), list, or None
        Smoothing parameter to pass to ``SeparateClassifiers``. Only used when passing to 'reward_estimator'
        a classifier with 'predict_proba'. See the documentation of ``SeparateClassifiers`` for details.
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores.
    kwargs_costsens
        Additional keyword arguments to pass to the cost-sensitive classifier.
    
    References
    ----------
    .. [1] Dudík, Miroslav, John Langford, and Lihong Li. "Doubly robust policy evaluation and learning."
           arXiv preprint arXiv:1103.4601 (2011).
    .. [2] Dudík, Miroslav, et al. "Doubly robust policy evaluation and optimization."
           Statistical Science 29.4 (2014): 485-511.
    """
    def __init__(self, base_algorithm, reward_estimator, nchoices, method='rovr',
                 handle_invalid=True, random_state=1, c=None, pmin=1e-5,
                 beta_prior=None, smoothing=(1.,2.), njobs=-1, **kwargs_costsens):
        assert (method == 'rovr') or (method == 'wap')
        self.method = method
        if method == 'wap':
            _check_constructor_input(base_algorithm, nchoices)
        else:
            assert isinstance(nchoices, int)
            assert nchoices > 2
            assert ('fit' in dir(base_algorithm)) and ('predict' in dir(base_algorithm))
        
        if c is not None:
            assert isinstance(c, float)
        if pmin is not None:
            assert isinstance(pmin, float)
        assert isinstance(handle_invalid, bool)
        
        if type(reward_estimator) == np.ndarray:
            assert reward_estimator.shape[1] == nchoices
            assert reward_estimator.shape[0] == X.shape[0]
        else:
            assert ('predict_proba_separate' in dir(reward_estimator)) or ('predict_proba' in dir(reward_estimator))

        if beta_prior is not None:
            beta_prior = _check_beta_prior(beta_prior, nchoices, 2)
        
        self.base_algorithm = base_algorithm
        self.reward_estimator = reward_estimator
        self.nchoices = nchoices
        self.c = c
        self.pmin = pmin
        self.handle_invalid = handle_invalid
        self.random_state = _check_random_state(random_state)
        self.beta_prior = beta_prior
        self.smoothing = _check_smoothing(smoothing, nchoices)
        self.njobs = _check_njobs(njobs)
        self.kwargs_costsens = kwargs_costsens
        self.is_fitted = False
    
    def fit(self, X, a, r, p):
        """
        Fits the Doubly-Robust estimator to partially-labeled data collected from a different policy.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.
        p : array (n_samples)
            Reward estimates for the actions that were chosen by the policy.
        """
        try:
            from costsensitive import RegressionOneVsRest, WeightedAllPairs
        except:
            raise ValueError("This functionality requires package 'costsensitive'.\nCan be installed with 'pip install costsensitive'.")
        p = _check_1d_inp(p)
        assert p.shape[0] == X.shape[0]
        l = -r
        
        if type(self.reward_estimator) == np.ndarray:
            C = self.reward_estimator
        elif 'predict_proba_separate' in dir(self.reward_estimator):
            C = -self.reward_estimator.predict_proba_separate(X)
        elif 'predict_proba' in dir(self.reward_estimator):
            reward_estimator = \
                SeparateClassifiers(self.reward_estimator, self.nchoices,
                    beta_prior = self.beta_prior, smoothing = self.smoothing,
                    random_state = self.random_state, njobs = self.njobs)
            reward_estimator.fit(X, a, r)
            C = -reward_estimator.predict_proba_separate(X)
        else:
            raise ValueError("Error: couldn't obtain reward estimates. Are you passing the right input to 'reward_estimator'?")
        
        if self.handle_invalid:
            C[C >= 1] = self.random_state.beta(3, 1, size = C.shape)[C >= 1]
            C[C <= 0] = self.random_state.beta(1, 3, size = C.shape)[C <= 0]
        
        if self.c is not None:
            p = self.c * p
        if self.pmin is not None:
            p = np.clip(p, a_min = self.pmin, a_max = None)
        
        C[np.arange(C.shape[0]), a] += (l - C[np.arange(C.shape[0]), a]) / p.reshape(-1)
        if self.method == 'rovr':
            self.oracle = RegressionOneVsRest(self.base_algorithm,
                                              njobs = self.njobs,
                                              **self.kwargs_costsens)
        else:
            self.oracle = WeightedAllPairs(self.base_algorithm,
                                           njobs = self.njobs,
                                           **self.kwargs_costsens)
        self.oracle.fit(X, C)
        self.is_fitted = True
    
    def predict(self, X):
        """
        Predict best arm for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action.
            
        Returns
        -------
        pred : array (n_samples,)
            Actions chosen by this technique.
        """
        assert self.is_fitted
        X = _check_X_input(X)
        return self.oracle.predict(X)
    
    def decision_function(self, X):
        """
        Get score distribution for the arm's rewards
        
        Note
        ----
        For details on how this is calculated, see the documentation of the
        RegressionOneVsRest and WeightedAllPairs classes in the costsensitive package.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to evaluate actions.
            
        Returns
        -------
        pred : array (n_samples, n_choices)
            Score assigned to each arm for each observation (see Note).
        """
        assert self.is_fitted
        X = _check_X_input(X)
        return self.oracle.decision_function(X)
    
    
class OffsetTree:
    """
    Offset Tree
    
    Parameters
    ----------
    base_algorithm : obj
        Binary classifier to be used for each classification sub-problem in the tree.
    nchoices : int
        Number of arms/labels to choose from.
    c : None or float
        Constant by which to multiply all scores from the exploration policy.
    pmin : None or float
        Scores (from the exploration policy) will be converted to the minimum between
        pmin and the original estimate.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        This is used when predictions need to be done for an arm with no data.
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Note that if the base algorithm is itself parallelized,
        this might result in a slowdown as both compete for available threads, so don't set
        parallelization in both.
    
    References
    ----------
    .. [1] Beygelzimer, Alina, and John Langford. "The offset tree for learning with partial labels."
           Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2009.
    """
    def __init__(self, base_algorithm, nchoices, c = None, pmin = 1e-5,
                 random_state = 1, njobs = -1):
        try:
            from costsensitive import _BinTree
        except:
            raise ValueError("This functionality requires package 'costsensitive'.\nCan be installed with 'pip install costsensitive'.")
        _check_constructor_input(base_algorithm, nchoices)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        self.tree = _BinTree(nchoices)
        if c is not None:
            assert isinstance(c, float)
        if pmin is not None:
            assert isinstance(pmin, float)
        self.c = c
        self.pmin = pmin
        self.random_state = _check_random_state(random_state)
        self.njobs = _check_njobs(njobs)
        self.is_fitted = False
    
    def fit(self, X, a, r, p):
        """
        Fits the Offset Tree estimator to partially-labeled data collected from a different policy.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.
        p : array (n_samples)
            Reward estimates for the actions that were chosen by the policy.
        """
        X, a, r = _check_fit_input(X, a, r)
        p = _check_1d_inp(p)
        assert p.shape[0] == X.shape[0]
        
        if self.c is not None:
            p = self.c * p
        if self.pmin is not None:
            p = np.clip(p, a_min = self.pmin, a_max = None)
        
        self._oracles = [deepcopy(self.base_algorithm) for c in range(self.nchoices - 1)]
        rs = self.random_state.integers(np.iinfo(np.int32).max, size=self.nchoices)
        Parallel(n_jobs=self.njobs, verbose=0, require="sharedmem")\
                (delayed(self._fit)(classif, X, a, r, p, rs) \
                    for classif in range(len(self._oracles)))
        self.is_fitted = True

    def _fit(self, classif, X, a, r, p, rs):
        obs_take = np.in1d(a, self.tree.node_comparisons[classif][0])
        X_node = X[obs_take, :]
        a_node = a[obs_take]
        r_node = r[obs_take]
        p_node = p[obs_take]
        
        r_more_onehalf = r_node >= .5
        y = (  np.in1d(a_node, self.tree.node_comparisons[classif][2])  ).astype('uint8')
        
        y_node = y.copy()
        y_node[r_more_onehalf] = 1. - y[r_more_onehalf]
        w_node = (.5 - r_node) / p_node
        w_node[r_more_onehalf] = (  (r_node - .5) / p_node  )[r_more_onehalf]
        w_node = w_node * (w_node.shape[0] / np.sum(w_node))
        
        n_pos = y_node.sum()
        if y_node.shape[0] == 0:
            self._oracles[classif] = _RandomPredictor(_check_random_state(rs[classif]))
        elif n_pos == y_node.shape[0]:
            self._oracles[classif] = _OnePredictor()
        elif n_pos == 0:
            self._oracles[classif] = _ZeroPredictor()
        else:
            self._oracles[classif].fit(X_node, y_node, sample_weight = w_node)
            
    def predict(self, X):
        """
        Predict best arm for new data.
        
        Note
        ----
        While in theory, making predictions from this algorithm should be faster than from others,
        the implementation here uses a Python loop for each observation, which is slow compared to
        NumPy array lookups, so the predictions will be slower to calculate than those from other algorithms.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action.
            
        Returns
        -------
        pred : array (n_samples,)
            Actions chosen by this technique.
        """
        assert self.is_fitted
        X = _check_X_input(X)
        pred = np.empty(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            pred[i] = self._predict(i, X)
        return pred

    def _predict(self, i, X):
        curr_node = 0
        X_this = X[i].reshape((1,-1))
        while True:
            go_right = self._oracles[curr_node].predict(X_this)
            if go_right:
                curr_node = self.tree.childs[curr_node][0]
            else:
                curr_node = self.tree.childs[curr_node][1]
                
            if curr_node <= 0:
                return -curr_node
