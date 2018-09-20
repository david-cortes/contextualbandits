from contextualbandits.utils import _check_constructor_input, _check_beta_prior, \
            _check_smoothing, _check_fit_input, _check_X_input, _check_1d_inp, \
            _BetaPredictor, _ZeroPredictor, _OnePredictor, _ArrBSClassif, _OneVsRest,\
            _calculate_beta_prior, _BayesianOneVsRest, _BayesianLogisticRegression,\
            _check_bools, _LinUCBSingle, _modify_predict_method, _check_active_inp, \
            _check_autograd_supported, _logistic_grad_pos, _logistic_grad_neg, _gen_random_grad
import warnings
import pandas as pd, numpy as np

class BootstrappedUCB:
    """
    Bootstrapped Upper Confidence Bound

    Obtains an upper confidence bound by taking the percentile of the predictions from a
    set of classifiers, all fit with different bootstrapped samples (multiple samples per arm).
    
    Note
    ----
    When fitting the algorithm to data in batches (online), it's not possible to take an
    exact bootstrapped sample, as the sample is not known in advance. In theory, as the sample size
    grows to infinity, the number of times that an observation appears in a bootstrapped sample is
    distributed ~ Poisson(1). However, I've found that assigning random weights to observations
    produces a more stable effect, so it also has the option to assign weights randomly ~ Gamma(1,1).
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
    nchoices : int
        Number of arms/labels to choose from.
    nsamples : int
        Number of bootstrapped samples per class to take.
    percentile : int [0,100]
        Percentile of the predictions sample to take
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive samples from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((5/nchoices, 4), 3)
        Note that it will only generate one random number per arm, so the 'a' parameters should be higher
        than for other methods.
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    smoothing : None or tuple (a,b)
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    batch_sample_method : str, either 'gamma' or 'poisson'
        How to simulate bootstrapped samples when training in batch mode (online).
        See Note.
        """
    def __init__(self, base_algorithm, nchoices, nsamples=10, percentile=80,
                 beta_prior='auto', smoothing=None, batch_train=False,
                 assume_unique_reward=False, batch_sample_method='gamma'):
        _check_constructor_input(base_algorithm, nchoices, batch_train)
        assert (percentile >= 0) and (percentile <= 100)
        assert (batch_sample_method == 'gamma') or (batch_sample_method == 'poisson')
        
        self.base_algorithm = base_algorithm
        if beta_prior=='auto':
            self.beta_prior = ((5/nchoices, 4), 3)
        else:
            self.beta_prior = _check_beta_prior(beta_prior, nchoices, 3)
        self.smoothing = _check_smoothing(smoothing)
        self.nchoices = nchoices
        self.nsamples = nsamples
        self.percentile = percentile
        
        self.batch_train, self.assume_unique_reward = _check_bools(batch_train, assume_unique_reward)
        self.batch_sample_method = batch_sample_method
    
    def fit(self, X, a, r):
        """
        Fits the base algorithm (one per sample per class) to partially labeled data,
        with the actions having been determined by this same policy.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.
        
        Returns
        -------
        self : obj
            Copy of this same object
        """
        X,a,r=_check_fit_input(X,a,r)
        self._oracles=_ArrBSClassif(self.base_algorithm,
                                   X,a,r,
                                   self.nchoices,
                                   self.beta_prior[1],self.beta_prior[0][0],self.beta_prior[0][1],
                                   self.nsamples,
                                   self.smoothing,
                                   self.assume_unique_reward,
                                   self.batch_train,
                                   self.batch_sample_method)
        return self
    
    def partial_fit(self, X, a, r):
        """
        Fits the base algorithm (one per sample per class) to partially labeled data in batches,
        with the actions having been determined by this same policy.
        
        Note
        ----
        In order to use this method, the base classifier must have a 'partial_fit' method,
        such as 'sklearn.linear_model.SGDClassifier'.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        if '_oracles' in dir(self):
            X,a,r=_check_fit_input(X,a,r)
            self._oracles.partial_fit(X,a,r)
            return self
        else:
            return self.fit(X,a,r)
    
    def decision_function(self, X):
        """
        Get the scores for each arm following this policy's action-choosing criteria.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to obtain decision function scores for each arm.
        
        Returns
        -------
        scores : array (n_samples, n_choices)
            Scores following this policy for each arm.
        """
        X=_check_X_input(X)
        return self._oracles.score_max(X,perc=self.percentile)
    
    def predict(self, X, exploit=False, output_score=False):
        """
        Selects actions according to this policy for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action according to this policy.
        exploit : bool
            Whether to make a prediction according to the policy, or to just choose the
            arm with the highest expected reward according to current models.
        output_score : bool
            Whether to output the score that this method predicted, in case it is desired to use
            it with this package's offpolicy and evaluation modules.
            
        Returns
        -------
        pred : array (n_samples,) or (n_samples, 2)
            Actions chosen by the policy. If passing output_score=True, it will be an array
            with the first column indicating the action and the second one indicating the score
            that the classifier gave to that class.
        """
        if exploit:
            X = _check_X_input(X)
            pred = self._oracles.score_avg(X)
        else:
            pred = self.decision_function(X)
            
        if not output_score:
            return np.argmax(pred, axis=1).reshape(-1)
        else:
            score_max = pred.max(axis=1).reshape(-1,1)
            pred = np.argmax(pred, axis=1).reshape(-1,1)
            return np.c_[pred, score_max]

class BootstrappedTS:
    """
    Bootstrapped Thompson Sampling
    
    Performs Thompson Sampling by fitting several models per class on bootstrapped samples,
    then makes predictions by taking one of them at random for each class.
    
    Note
    ----
    When fitting the algorithm to data in batches (online), it's not possible to take an
    exact bootstrapped sample, as the sample is not known in advance. In theory, as the sample size
    grows to infinity, the number of times that an observation appears in a bootstrapped sample is
    distributed ~ Poisson(1). However, I've found that assigning random weights to observations
    produces a more stable effect, so it also has the option to assign weights randomly ~ Gamma(1,1).
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
    nchoices : int
        Number of arms/labels to choose from.
    nsamples : int
        Number of bootstrapped samples per class to take.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive samples from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices, 4), 3)
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    smoothing : None or tuple (a,b)
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    batch_sample_method : str, either 'gamma' or 'poisson'
        How to simulate bootstrapped samples when training in batch mode (online).
        See Note.
    
    References
    ----------
    [1] An empirical evaluation of thompson sampling (2011)
    """
    def __init__(self, base_algorithm, nchoices, nsamples=10, beta_prior='auto', smoothing=None,
                 batch_train=False, assume_unique_reward=False, batch_sample_method='gamma'):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        assert isinstance(nsamples, int)
        assert nsamples>=2
        assert (batch_sample_method=='gamma') or (batch_sample_method=='poisson')
        
        self.beta_prior = _check_beta_prior(beta_prior, nchoices, 3)
        self.smoothing = _check_smoothing(smoothing)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        self.nsamples=nsamples
        
        self.batch_train, self.assume_unique_reward = _check_bools(batch_train, assume_unique_reward)
        self.batch_sample_method = batch_sample_method
    
    def fit(self, X, a, r):
        """
        Fits the base algorithm (one per sample per class) to partially labeled data,
        with the actions having been determined by this same policy.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        X,a,r=_check_fit_input(X,a,r)
        self._oracles=_ArrBSClassif(self.base_algorithm,
                                   X,a,r,
                                   self.nchoices,
                                   self.beta_prior[1],self.beta_prior[0][0],self.beta_prior[0][1],
                                   self.nsamples,
                                   self.smoothing,
                                   self.assume_unique_reward,
                                   self.batch_train,
                                   self.batch_sample_method)
        return self
    
    def partial_fit(self, X, a, r):
        """
        Fits the base algorithm (one per class) to partially labeled data in batches,
        with the actions having been determined by this same policy.
        
        Note
        ----
        In order to use this method, the base classifier must have a 'partial_fit' method,
        such as 'sklearn.linear_model.SGDClassifier'.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        if '_oracles' in dir(self):
            X,a,r=_check_fit_input(X,a,r)
            self._oracles.partial_fit(X,a,r)
            return self
        else:
            return self.fit(X,a,r)
    
    def decision_function(self, X):
        """
        Get the scores for each arm following this policy's action-choosing criteria.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to obtain decision function scores for each arm.
        
        Returns
        -------
        scores : array (n_samples, n_choices)
            Scores following this policy for each arm.
        """
        X=_check_X_input(X)
        return self._oracles.score_rnd(X)
    
    def predict(self, X, exploit=False, output_score=False):
        """
        Selects actions according to this policy for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action according to this policy.
        exploit : bool
            Whether to make a prediction according to the policy, or to just choose the
            arm with the highest expected reward according to current models.
        output_score : bool
            Whether to output the score that this method predicted, in case it is desired to use
            it with this pakckage's offpolicy and evaluation modules.
            
        Returns
        -------
        pred : array (n_samples,) or (n_samples, 2)
            Actions chosen by the policy. If passing output_score=True, it will be an array
            with the first column indicating the action and the second one indicating the score
            that the classifier gave to that class.
        """
        if exploit:
            X = _check_X_input(X)
            pred = self._oracles.score_avg(X)
        else:
            pred = self.decision_function(X)
        
        if not output_score:
            return np.argmax(pred, axis=1).reshape(-1)
        else:
            score_max = pred.max(axis=1).reshape(-1,1)
            pred = np.argmax(pred, axis=1).reshape(-1,1)
            return np.c_[pred, score_max]

class SeparateClassifiers:
    """
    Separate Clasifiers per arm
    
    Fits one classifier per arm using only the data on which that arm was chosen.
    Predicts as One-Vs-Rest.
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
    nchoices : int
        Number of arms/labels to choose from.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive samples from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices, 4), 3)
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    smoothing : None or tuple (a,b)
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    """
    def __init__(self, base_algorithm, nchoices, beta_prior=None, smoothing=None,
                 batch_train=False, assume_unique_reward=False):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        self.beta_prior = _check_beta_prior(beta_prior, nchoices, 3)
        self.smoothing = _check_smoothing(smoothing)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        
        self.batch_train, self.assume_unique_reward = _check_bools(batch_train, assume_unique_reward)
    
    def fit(self, X, a, r):
        """
        Fits the base algorithm (one per class) to partially labeled data.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        X,a,r=_check_fit_input(X,a,r)
        self._oracles=_OneVsRest(self.base_algorithm,
                                   X,a,r,
                                   self.nchoices,
                                   self.beta_prior[1],self.beta_prior[0][0],self.beta_prior[0][1],
                                   self.smoothing,
                                   self.assume_unique_reward,
                                   self.batch_train)
        return self
    
    def partial_fit(self, X, a, r):
        """
        Fits the base algorithm (one per class) to partially labeled data in batches.
        
        Note
        ----
        In order to use this method, the base classifier must have a 'partial_fit' method,
        such as 'sklearn.linear_model.SGDClassifier'.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        if '_oracles' in dir(self):
            X,a,r=_check_fit_input(X,a,r)
            self._oracles.partial_fit(X,a,r)
            return self
        else:
            return self.fit(X,a,r)
    
    def decision_function(self,X):
        """
        Get the scores for each arm following this policy's action-choosing criteria.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to obtain decision function scores for each arm.
        
        Returns
        -------
        scores : array (n_samples, n_choices)
            Scores following this policy for each arm.
        """
        X=_check_X_input(X)
        return self._oracles.decision_function(X)
    
    def decision_function_std(self,X):
        """
        Get the predicted probabilities from each arm from the classifier that predicts it,
        standardized to sum up to 1.
        
        Note
        ----
        Classifiers are all fit on different data, so the probabilities will not add up to 1.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to obtain decision function scores for each arm.
        
        Returns
        -------
        scores : array (n_samples, n_choices)
            Scores following this policy for each arm.
        """
        X=_check_X_input(X)
        return self._oracles.predict_proba(X)
    
    def predict_proba_separate(self,X):
        """
        Get the predicted probabilities from each arm from the classifier that predicts it.
        
        Note
        ----
        Classifiers are all fit on different data, so the probabilities will not add up to 1.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to obtain decision function scores for each arm.
        
        Returns
        -------
        scores : array (n_samples, n_choices)
            Scores following this policy for each arm.
        """
        X=_check_X_input(X)
        return self._oracles.predict_proba_raw(X)
    
    def predict(self, X, exploit=False, output_score=False):
        """
        Selects actions according to this policy for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action according to this policy.
        exploit : bool
            Whether to make a prediction according to the policy, or to just choose the
            arm with the highest expected reward according to current models.
        output_score : bool
            Whether to output the score that this method predicted, in case it is desired to use
            it with this pakckage's offpolicy and evaluation modules.
            
        Returns
        -------
        pred : array (n_samples,) or (n_samples, 2)
            Actions chosen by the policy. If passing output_score=True, it will be an array
            with the first column indicating the action and the second one indicating the score
            that the classifier gave to that class.
        """
        scores = self.decision_function(X)
        pred = np.argmax(scores, axis=1)
        if not output_score:
            return pred
        else:
            score_max = np.max(scores, axis=1).reshape(-1,1)
            return np.c_[pred.reshape(-1,1), score_max]

class EpsilonGreedy:
    """
    Epsilon Greedy
    
    Takes a random action with probability p, or the action with highest
    estimated reward with probability 1-p.
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
    nchoices : int
        Number of arms/labels to choose from.
    explore_prob : float (0,1)
        Probability of taking a random action at each round.
    decay : float (0,1)
        After each prediction, the explore probability reduces to
        p = p*decay
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive samples from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices, 4), 3)
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    smoothing : None or tuple (a,b)
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    
    References
    ----------
    [1] The k-armed dueling bandits problem (2010)
    """
    def __init__(self, base_algorithm, nchoices, explore_prob=0.2, decay=0.9999,
                 beta_prior='auto', smoothing=None, batch_train=False, assume_unique_reward=False):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        self.beta_prior = _check_beta_prior(beta_prior, nchoices, 3)
        self.smoothing = _check_smoothing(smoothing)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        
        assert (explore_prob>0) and (explore_prob<1)
        if decay is not None:
            assert (decay>0) and (decay<1)
            if decay<=.99:
                warnings.warn("Warning: 'EpsilonGreedy' has a very high decay rate.")
        self.explore_prob = explore_prob
        self.decay = decay
        
        self.batch_train, self.assume_unique_reward = _check_bools(batch_train, assume_unique_reward)
    
    def fit(self, X, a, r):
        """
        Fits the base algorithm (one per class) to partially labeled data.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        X,a,r=_check_fit_input(X,a,r)
        self._oracles=_OneVsRest(self.base_algorithm,
                                   X,a,r,
                                   self.nchoices,
                                   self.beta_prior[1],self.beta_prior[0][0],self.beta_prior[0][1],
                                   self.smoothing,
                                   self.assume_unique_reward,
                                   self.batch_train)
        return self
    
    def partial_fit(self, X, a, r):
        """
        Fits the base algorithm (one per class) to partially labeled data in batches.
        
        Note
        ----
        In order to use this method, the base classifier must have a 'partial_fit' method,
        such as 'sklearn.linear_model.SGDClassifier'.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        if '_oracles' in dir(self):
            X,a,r=_check_fit_input(X,a,r)
            self._oracles.partial_fit(X,a,r)
            return self
        else:
            return self.fit(X,a,r)
    
    def decision_function(self,X):
        """
        Get the decision function for each arm from the classifier that predicts it.
        
        Note
        ----
        This is quite different from the decision_function of the other policies, as it
        doesn't follow the policy in assigning random choices with some probability.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to obtain decision function scores for each arm.
        
        Returns
        -------
        scores : array (n_samples, n_choices)
            Scores following this policy for each arm.
        """
        X=_check_X_input(X)
        return self._oracles.decision_function(X)
    
    def predict(self, X, exploit=False, output_score=False):
        """
        Selects actions according to this policy for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action according to this policy.
        exploit : bool
            Whether to make a prediction according to the policy, or to just choose the
            arm with the highest expected reward according to current models.
        output_score : bool
            Whether to output the score that this method predicted, in case it is desired to use
            it with this pakckage's offpolicy and evaluation modules.
            
        Returns
        -------
        pred : array (n_samples,) or (n_samples, 2)
            Actions chosen by the policy. If passing output_score=True, it will be an array
            with the first column indicating the action and the second one indicating the score
            that the classifier gave to that class.
        """
        scores = self.decision_function(X)
        pred = np.argmax(scores, axis=1)
        if not exploit:
            ix_change_rnd = (np.random.random(size =  X.shape[0]) <= self.explore_prob)
            pred[ix_change_rnd] = np.random.randint(self.nchoices, size=ix_change_rnd.sum())
        if self.decay is not None:
            self.explore_prob *= self.decay**X.shape[0]
        if not output_score:
            return pred
        else:
            score_max = np.max(scores, axis=1).reshape(-1,1)
            score_max[ix_change_rnd] = 1 / self.nchoices
            return np.c_[pred.reshape(-1,1), score_max]


class AdaptiveGreedy:
    """
    Adaptive Greedy
    
    Takes the action with highest estimated reward, unless that estimation falls below a certain
    threshold, in which case it takes a random action or some other action according to
    an active learning heuristic.
    
    Note
    ----
    The threshold for the reward probabilities can be set to a hard-coded number, or
    to be calculated dynamically by keeping track of the predictions it makes, and taking
    a fixed percentile of that distribution to be the threshold.
    In the second case, these are calculated in separate batches rather than in a sliding window.
    
    The original idea was taken from the paper in the references and adapted to the
    contextual bandits setting like this. Can also be set to make choices in the same way as
    'ActiveExplorer' rather than random (see 'greedy_choice' parameter) when using logistic regression.
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
    nchoices : int
        Number of arms/labels to choose from.
    window_size : int
        Number of predictions after which the threshold will be updated to the desired percentile.
        Ignored when passing fixed_thr=False
    percentile : int [0,100]
        Percentile of the predictions sample to set as threshold, below which actions are random.
        Ignored in fixed threshold mode.
    decay : float (0,1)
        After each prediction, either the threshold or the percentile gets adjusted to:
            val_t+1 = val_t*decay
        Ignored when pasing fixed_thr=True.
    decay_type : str, either 'percentile' or 'threshold'
        Whether to decay the threshold itself or the percentile of the predictions to take after
        each prediction. If set to 'threshold' and fixed_thr=False, the threshold will be
        recalculated to the same percentile the next time it is updated, but with the latest predictions.
        Ignored when passing fixed_thr=True.
    initial_thr : str 'autho' or float (0,1)
        Initial threshold for the prediction below which a random action is taken.
        If set to 'auto', will be calculated as initial_thr = 1.5/nchoices
    fixed_thr : bool
        Whether the threshold is to be kept fixed, or updated to a percentile after N predictions.
    greedy_choice : None or str in {'min', 'max', 'weighted'}
        How to select arms when predictions are below the threshold. If passing None, selects them at random.
        If passing 'min', 'max' or 'weighted', selects them in the same way as 'ActiveExplorer'.
        Non-random selection requires classifiers with a 'coef_' attribute, such as logistic regression.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive samples from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices, 4), 3)
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    smoothing : None or tuple (a,b)
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    
    References
    ----------
    [1] Mortal multi-armed bandits (2009)
    
    """
    def __init__(self, base_algorithm, nchoices, window_size=500, percentile=30, decay=0.9998,
                     decay_type='percentile', initial_thr='auto', fixed_thr=False,
                     beta_prior='auto', smoothing=None, batch_train=False, assume_unique_reward=False,
                     active_choice=None, grad_pos='auto', grad_neg='auto', case_one_class='auto'):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        self.beta_prior = _check_beta_prior(beta_prior, nchoices, 3)
        self.smoothing = _check_smoothing(smoothing)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        
        assert isinstance(window_size, int)
        assert isinstance(percentile, int)
        if initial_thr == 'auto':
            initial_thr = 1 / (nchoices / 1.5)
        assert isinstance(initial_thr, float)
        assert (percentile > 0) and (percentile < 100)
        assert window_size > 0
        assert isinstance(fixed_thr, bool)
        self.window_size = window_size
        self.percentile = percentile
        self.thr = initial_thr
        self.fixed_thr = fixed_thr
        self.window_cnt = 0
        self.window = np.array([])
        self.decay = decay
        assert (decay_type == 'threshold') or (decay_type == 'percentile')
        self.decay_type = decay_type
        
        self.batch_train, self.assume_unique_reward = _check_bools(batch_train, assume_unique_reward)

        if active_choice is not None:
            assert active_choice in ['min', 'max', 'weighted']
            self.active_choice = active_choice
            _check_active_inp(self, grad_pos, grad_neg, case_one_class)
        else:
            self._force_fit = False
    
    def fit(self, X, a, r):
        """
        Fits the base algorithm (one per class) to partially labeled data.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        X,a,r=_check_fit_input(X,a,r)
        self._oracles=_OneVsRest(self.base_algorithm,
                                   X,a,r,
                                   self.nchoices,
                                   self.beta_prior[1],self.beta_prior[0][0],self.beta_prior[0][1],
                                   self.smoothing,
                                   self.assume_unique_reward,
                                   self.batch_train,
                                   self._force_fit)
        return self
    
    def partial_fit(self, X, a, r):
        """
        Fits the base algorithm (one per class) to partially labeled data in batches.
        
        Note
        ----
        In order to use this method, the base classifier must have a 'partial_fit' method,
        such as 'sklearn.linear_model.SGDClassifier'.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        if '_oracles' in dir(self):
            X,a,r=_check_fit_input(X,a,r)
            self._oracles.partial_fit(X,a,r)
            return self
        else:
            return self.fit(X,a,r)
    
    def predict(self, X, exploit=False):
        """
        Selects actions according to this policy for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action according to this policy.
        exploit : bool
            Whether to make a prediction according to the policy, or to just choose the
            arm with the highest expected reward according to current models.
            
        Returns
        -------
        pred : array (n_samples,)
            Actions chosen by the policy.
        """
        # TODO: add option to output scores
        X = _check_X_input(X)
        
        if X.shape[0] == 0:
            return np.array([])
        
        if exploit:
            return self._oracles.predict(X)
        
        # fixed threshold, anything below is always random
        if self.fixed_thr:
            pred, pred_max = self._calc_preds(X)
        else:
            remainder_window = self.window_size - self.window_cnt
            
            # case 1: number of predictions to make would still fit within current window
            if remainder_window >= X.shape[0]:
                pred, pred_max = self._calc_preds(X)
                self.window_cnt += X.shape[0]
                self.window = np.r_[self.window, pred_max]
                
                # apply decay for all observations
                self._apply_decay(self, X.shape[0])

            # case 2: number of predictions to make would span more than current window
            else:
                # predict for the remainder of this window
                pred, pred_max = self._calc_preds(X[:remainder_window, :])
                
                # allocate the rest that don't fit in this window
                pred_all = np.zeros(X.shape[0])
                pred_all[:remainder_window] = pred
                
                # complete window, update percentile if needed
                self.window = np.r_[self.window, pred_max]
                if self.decay_type == 'percentile':
                    self.thr = np.percentile(self.window, self.percentile)

                # reset window
                self.window = np.array([])
                self.window_cnt = 0
                
                # decay threshold only for these observations
                self._apply_decay(self, remainder_window)
                
                # predict the rest recursively
                pred_all[remainder_window:] = self.predict(X[remainder_window:, :])
                return pred_all
                
        return pred

    def _apply_decay(self, nobs):
        if self.decay is not None:
            if self.decay_type == 'threshold':
                self.thr *= self.decay ** nobs
            elif self.decay_type == 'percentile':
                self.percentile *= self.decay ** nobs
            else:
                raise ValueError("'decay_type' must be one of 'threshold', 'percentile', or None.")

    def _calc_preds(self, X):
        pred_proba = self._oracles.predict_proba(X)
        pred_max = pred_proba.max(axis=1)
        pred = np.argmax(pred_proba, axis=1)
        set_greedy = pred_max <= self.thr
        self._choose_greedy(set_greedy, X, pred, pred_proba)
        return pred, pred_max

    def _choose_greedy(self, set_greedy, X, pred, pred_all):
        if self.active_choice is None:
            pred[set_greedy] = np.random.randint(self.nchoices, size = set_greedy.sum())
        else:
            ActiveExplorer._crit_active(self, X[set_greedy], pred_all[set_greedy], self.active_choice)
            pred[set_greedy] = np.argmax(pred_all[set_greedy], axis = 1)
    
    def decision_function(self, X):
        """
        Get the estimated probability for each arm from the classifier that predicts it.
        
        Note
        ----
        This is quite different from the decision_function of the other policies, as it
        doesn't follow the policy in assigning random choices with some probability.
        A sigmoid function is applyed to the decision_function of the classifier if it doesn't
        have a predict_proba method.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to obtain decision function scores for each arm.
        
        Returns
        -------
        scores : array (n_samples, n_choices)
            Scores following this policy for each arm.
        """
        X=_check_X_input(X)
        return self._oracles.predict_proba(X)

class ExploreFirst:
    """
    Explore First, a.k.a. Explore-Then-Exploit
    
    Selects random actions for the first N predictions, after which it selects the
    best arm only according to its estimates.
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
    nchoices : int
        Number of arms/labels to choose from.
    explore_rounds : int
        Number of rounds to wait before exploitation mode.
        Will switch after making N predictions.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive samples from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices, 4), 3)
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    smoothing : None or tuple (a,b)
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    """
    def __init__(self, base_algorithm, nchoices, explore_rounds=2500,
                 beta_prior='auto',smoothing=None, batch_train=False, assume_unique_reward=False):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        self.beta_prior = _check_beta_prior(beta_prior, nchoices, 3)
        self.smoothing = _check_smoothing(smoothing)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        
        assert explore_rounds>0
        assert isinstance(explore_rounds, int)
        self.explore_rounds = explore_rounds
        self.explore_cnt = 0
        
        self.batch_train, self.assume_unique_reward = _check_bools(batch_train, assume_unique_reward)
    
    def fit(self, X, a, r):
        """
        Fits the base algorithm (one per class) to partially labeled data with actions chosen by this same policy.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        X,a,r=_check_fit_input(X,a,r)
        self._oracles=_OneVsRest(self.base_algorithm,
                                   X,a,r,
                                   self.nchoices,
                                   self.beta_prior[1],self.beta_prior[0][0],self.beta_prior[0][1],
                                   self.smoothing,
                                   self.assume_unique_reward,
                                   self.batch_train)
        return self
    
    def partial_fit(self, X, a, r):
        """
        Fits the base algorithm (one per class) to partially labeled data in batches.
        
        Note
        ----
        In order to use this method, the base classifier must have a 'partial_fit' method,
        such as 'sklearn.linear_model.SGDClassifier'.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        if '_oracles' in dir(self):
            X,a,r=_check_fit_input(X,a,r)
            self._oracles.partial_fit(X,a,r)
            return self
        else:
            return self.fit(X,a,r)
    
    def predict(self, X, exploit=False):
        """
        Selects actions according to this policy for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action according to this policy.
        exploit : bool
            Whether to make a prediction according to the policy, or to just choose the
            arm with the highest expected reward according to current models.
            
        Returns
        -------
        pred : array (n_samples,)
            Actions chosen by the policy.
        """
        # TODO: add option to output scores
        X=_check_X_input(X)
        
        if X.shape[0]==0:
            return np.array([])
        
        if exploit:
            return self._oracles.predict(X)
        
        if self.explore_cnt<self.explore_rounds:
            self.explore_cnt+=X.shape[0]
            
            # case 1: all predictions are within allowance
            if self.explore_cnt<=self.explore_rounds:
                return np.random.randint(self.nchoices, size=X.shape[0])
            
            # case 2: some predictions are within allowance, others are not
            else:
                nexplore=self.explore_rounds-self.explore_cnt
                pred=np.zeros(X.shape[0])
                pred[:nexplore]=np.random.randint(self.nchoices, nexplore)
                pred[nexplore:]=self.predict(X)
                return pred
        else:
            return self._oracles.predict(X).reshape(-1)
        
    def decision_function(self, X):
        """
        Get the decision function for each arm from the classifier that predicts it.
        
        Note
        ----
        This is quite different from the decision_function of the other policies, as it
        doesn't follow the policy in assigning random choices at the beginning with equal
        probability all.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to obtain decision function scores for each arm.
        
        Returns
        -------
        scores : array (n_samples, n_choices)
            Scores following this policy for each arm.
        """
        X=_check_X_input(X)
        return self._oracles.predict_proba(X)

class ActiveExplorer:
    """
    Active Explorer
    
    Selects a proportion of actions according to an active learning heuristic based on gradient.
    Works only for differentiable and preferably smooth functions.
    
    Note
    ----
    Here, for the predictions that are made according to an active learning heuristic
    (these are selected at random, just like in Epsilon-Greedy), the guiding heuristic
    is the gradient that the observation, having either label (either weighted by the estimted
    probability, or taking the maximum or minimum), would produce on each model that
    predicts a class, given the current coefficients for that model.
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
    grad_pos : str 'auto' or function(base_algorithm, X, pred) -> array (n_samples,)
        Function that calculates the row-wise norm of the gradient of observations in X if their label were positive.
        The option 'auto' will only work with scikit-learn's 'LogisticRegression', 'SGDClassifier', and 'RidgeClassifier'.
    grad_neg : str 'auto' or function(base_algorithm, X, pred) -> array (n_samples,)
        Function that calculates the row-wise norm of the gradient of observations in X if their label were negative.
        The option 'auto' will only work with scikit-learn's 'LogisticRegression', 'SGDClassifier', and 'RidgeClassifier'.
    case_one_class : str 'auto', None, or function(X) -> array(n_samples,)
        If some arm/choice/class has only rewards of one type, many models will fail to fit, and consequently the gradients
        will be undefined. Likewise, if the model has not been fit, the gradient might also be undefined, and this requires a workaround.
        If passing None, will assume that 'base_algorithm' can be fit to data of only-positive or only-negative class without
        problems, that the gradients can be calculated with these fitted models, and that it can calculate gradients and predictions
        with a 'base_algorithm' object that has not been fitted.
        If passing a function, will take the output of it as the gradients when it compares them against other arms/classes.
        If passing 'auto', will generate random numbers ~ Gamma(sqrt(n_features), 1).
    nchoices : int
        Number of arms/labels to choose from.
    explore_prob : float (0,1)
        Probability of selecting an action according to active learning criteria.
    decay : float (0,1)
        After each prediction, the probability of selecting an arm according to active
        learning criteria is set to p = p*decay
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive samples from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices, 4), 3)
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    smoothing : None or tuple (a,b)
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_seed : None or int
        Random state or seed to pass to the solver.
    """
    def __init__(self, base_algorithm, nchoices, grad_pos='auto', grad_neg='auto',
                 explore_prob=.15, decay=0.9997, beta_prior='auto', smoothing=None,
                 batch_train=False, assume_unique_reward=False, random_seed=None):
        _check_active_inp(self, grad_pos, grad_neg, case_one_class)
        _check_constructor_input(base_algorithm, nchoices, batch_train)
        self.beta_prior = _check_beta_prior(beta_prior, nchoices, 3)
        self.smoothing = _check_smoothing(smoothing)

        if batch_train:
            base_algorithm = _modify_predict_method(base_algorithm)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        
        assert isinstance(explore_prob, float)
        assert (explore_prob > 0) and (explore_prob < 1)
        self.explore_prob = explore_prob
        self.decay = decay
        
        self.batch_train, self.assume_unique_reward = _check_bools(batch_train, assume_unique_reward)
    
    def fit(self, X, a, r):
        """
        Fits logistic regression (one per class) to partially labeled data,
        with actions chosen by this same policy.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        self._oracles=_OneVsRest(self.base_algorithm,
                                   X,a,r,
                                   self.nchoices,
                                   self.beta_prior[1],self.beta_prior[0][0],self.beta_prior[0][1],
                                   self.smoothing,
                                   self.assume_unique_reward,
                                   self.batch_train,
                                   self._force_fit)
        return self
    
    def partial_fit(self, X, a, r):
        """
        Fits logistic regression (one per class) to partially labeled data in batches,
        with actions chosen by this same policy.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        if '_oracles' in dir(self):
            X,a,r=_check_fit_input(X,a,r)
            self._oracles.partial_fit(X,a,r)
            return self
        else:
            return self.fit(X,a,r)
            
    
    def predict(self, X, exploit=False, gradient_calc='weighted'):
        """
        Selects actions according to this policy for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action according to this policy.
        exploit : bool
            Whether to make a prediction according to the policy, or to just choose the
            arm with the highest expected reward according to current models.
        gradient_calc : str, one of 'weighted', 'max' or 'min'
            How to calculate the gradient that an observation would have on the loss
            function for each classifier, given that it could be either class (positive or negative)
            for the classifier that predicts each arm. If weighted, they are weighted by the same
            probability estimates from the base algorithm.
            
        Returns
        -------
        pred : array (n_samples,)
            Actions chosen by the policy.
        """
        X = _check_X_input(X)
        
        pred = self._oracles.decision_function(X)
        if not exploit:
            change_greedy = np.random.random(size=X.shape[0]) <= self.explore_prob
            self._crit_active(X[change_greedy, :], pred[change_greedy, :], gradient_calc)
            
            if self.decay is not None:
                self.explore_prob *= self.decay ** X.shape[0]
        
        return np.argmax(pred, axis=1)

    def _crit_active(self, X, pred, grad_crit):
        for choice in range(self.nchoices):
            if self.oracles.algos[choice].__class__.__name__ in ['_BetaPredictor', '_OnePredictor', '_ZeroPredictor']:
                grad_both = np.c_[self._rand_grad(X), self._rand_grad(X)]
            else:
                grad_both = np.c_[self.grad_pos(self.oracles.algos[choice], X, pred[:, choice]), self.grad_neg(self.oracles.algos[choice], X, pred[:, choice])]

            if grad_crit == 'min':
                pred[:, choice] = grad_both.min(axis = 1)
            elif grad_crit == 'max':
                pred[:, choice] = grad_both.max(axis = 1)
            elif grad_crit == 'weighted':
                pred[:, choice] = (pred * grad_both).sum(axis = 1)
            else:
                raise ValueError("Something went wrong. Please open an issue in GitHub indicating what you were doing.")

class SoftmaxExplorer:
    """
    SoftMax Explorer
    
    Selects an action according to probabilites determined by a softmax transformation
    on the scores from the decision function that predicts each class.
    
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1, to which it
               will apply an inverse sigmoid function.
            2) A 'decision_function' method with unbounded outputs (n_samples,).
            3) A 'predict' method outputting (n_samples,), values in [0,1], to which it will apply an inverse sigmoid function.
    nchoices : int
        Number of arms/labels to choose from.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive samples from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices, 4), 3)
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    smoothing : None or tuple (a,b)
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of 'beta_prior' or 'smoothing'.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    """
    def __init__(self, base_algorithm, nchoices, beta_prior='auto', smoothing=None,
                 batch_train=False, assume_unique_reward=False):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        self.beta_prior = _check_beta_prior(beta_prior, nchoices, 3)
        self.smoothing = _check_smoothing(smoothing)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        
        self.batch_train, self.assume_unique_reward = _check_bools(batch_train, assume_unique_reward)
    
    def fit(self, X, a, r):
        """
        Fits the base algorithm (one per class) to partially labeled data.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        X,a,r=_check_fit_input(X,a,r)
        self._oracles=_OneVsRest(self.base_algorithm,
                                   X,a,r,
                                   self.nchoices,
                                   self.beta_prior[1],self.beta_prior[0][0],self.beta_prior[0][1],
                                   self.smoothing,
                                   self.assume_unique_reward,
                                   self.batch_train)
        return self
    
    def partial_fit(self, X, a, r):
        """
        Fits the base algorithm (one per class) to partially labeled data in batches.
        
        Note
        ----
        In order to use this method, the base classifier must have a 'partial_fit' method,
        such as 'sklearn.linear_model.SGDClassifier'.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        if '_oracles' in dir(self):
            X,a,r=_check_fit_input(X,a,r)
            self._oracles.partial_fit(X,a,r)
            return self
        else:
            return self.fit(X,a,r)
    
    def decision_function(self, X, output_score=False, apply_sigmoid_score=True):
        """
        Get the scores for each arm following this policy's action-choosing criteria.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to obtain decision function scores for each arm.
        
        Returns
        -------
        scores : array (n_samples, n_choices)
            Scores following this policy for each arm.
        """
        X=_check_X_input(X)
        return self._oracles.predict_proba(X)
    
    def predict(self, X, exploit=False, output_score=False):
        """
        Selects actions according to this policy for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action according to this policy.
        exploit : bool
            Whether to make a prediction according to the policy, or to just choose the
            arm with the highest expected reward according to current models.
        output_score : bool
            Whether to output the score that this method predicted, in case it is desired to use
            it with this pakckage's offpolicy and evaluation modules.
            
        Returns
        -------
        pred : array (n_samples,) or (n_samples, 2)
            Actions chosen by the policy. If passing output_score=True, it will be an array
            with the first column indicating the action and the second one indicating the score
            that the classifier gave to that class.
        """
        if exploit:
            X=_check_X_input(X)
            return self._oracles.predict(X)
        pred=self.decision_function(X)
        chosen=list()
        for p_arr in pred:
            if np.sum(p_arr)!=1.0:
                p_arr=p_arr/np.sum(p_arr)
            chosen.append(np.random.choice(self.nchoices, p=p_arr))
        if not output_score:
            return np.array(chosen)
        else:
            return np.c_[np.array(chosen).reshape(-1,1),\
                         pred[np.arange(pred.shape[0]),np.array(chosen)].reshape(-1,1)]


class LinUCB:
    """
    LinUCB
    
    Note
    ----
    The formula here is implemented in a loop per observation for both fitting and predicting.
    The A matrix in the formulas are not inverted after each updte, but rather, only their inverse is stored
    and is updated after each observation using the Sherman-Morrison formula.
    Thus, updating is quite fast, but there is no speed-up in doing batch updates.
    
    Parameters
    ----------
    nchoices : int
        Number of arms/labels to choose from.
    alpha : float
        Parameter to control the upper confidence bound (more is higher).
    
    References
    ----------
    [1] A contextual-bandit approach to personalized news article recommendation (2010)
    """
    def __init__(self, nchoices, alpha=1.0):
        if isinstance(alpha, int):
            alpha = float(alpha)
        assert isinstance(alpha, float)
        _check_constructor_input(_ZeroPredictor(), nchoices)
        self.alpha = alpha
        self.nchoices = nchoices
        self._oracles = [_LinUCBSingle(self.alpha) for n in range(nchoices)]
    
    def fit(self, X, a, r):
        """"
        Fits one linear model for the first time to partially labeled data.
        Overwrites previously fitted coefficients if there were any.
        (See partial_fit for adding more data in batches)

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        X,a,r=_check_fit_input(X,a,r)
        self.ndim = X.shape[1]
        for n in range(self.nchoices):
            this_action=a==n
            self._oracles[n].fit(X[this_action,:],r[this_action].astype('float64'))

        return self
                
    def partial_fit(self, X, a, r):
        """"
        Updates each linear model with a new batch of data with actions chosen by this same policy.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        X,a,r=_check_fit_input(X,a,r)
        for n in range(self.nchoices):
            this_action=a==n
            self._oracles[n].partial_fit(X[this_action,:], r[this_action].astype('float64'))
            
        return self
    
    def predict(self, X, exploit=False):
        """
        Selects actions according to this policy for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action according to this policy.
        exploit : bool
            Whether to make a prediction according to the policy, or to just choose the
            arm with the highest expected reward according to current models.
            
        Returns
        -------
        pred : array (n_samples,)
            Actions chosen by the policy.
        """
        X=_check_X_input(X)
        pred=np.zeros((X.shape[0],self.nchoices))
        for choice in range(self.nchoices):
            pred[:,choice]=self._oracles[choice].predict(X)
        return np.argmax(pred, axis=1)

class BayesianUCB:
    """
    Bayesian Upper Confidence Bound
    
    Gets an upper confidence bound by Bayesian Logistic Regression estimates.
    
    Note
    ----
    The implementation here uses PyMC3's GLM formula with default parameters and ADVI.
    This is a very, very slow implementation, and will probably take at least two
    orders or magnitude more to fit compared to other methods.
    
    Parameters
    ----------
    nchoices : int
        Number of arms/labels to choose from.
    percentile : int [0,100]
        Percentile of the predictions sample to take
    method : str, either 'advi' or 'nuts'
        Method used to sample coefficients (see PyMC3's documentation for mode details).
     beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive samples from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((5/nchoices, 4), 3)
        Note that it will only generate one random number per arm, so the 'a' parameters should be higher
        than for other methods.
    """
    def __init__(self, nchoices, percentile=80, method='advi', nsamples=None, beta_prior='auto'):

        ## NOTE: this is a really slow and poorly thought implementation
        ## TODO: rewrite using some faster framework such as Edward,
        ##       or with a hard-coded coordinate ascent procedure instead. 

        import pymc3 as pm
        if beta_prior=='auto':
            self.beta_prior = ((5/nchoices, 4), 3)
        else:
            self.beta_prior = _check_beta_prior(beta_prior, nchoices, 3)
        _check_constructor_input(_ZeroPredictor(), nchoices)
        self.nchoices = nchoices
        assert method in ['advi','nuts']
        assert (percentile >= 0) and (percentile <= 100)
        self.method = method
        self.percentile = percentile
    
    def fit(self, X, a, r):
        """
        Samples Logistic Regression coefficients for partially labeled data with actions chosen by this policy.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        X,a,r=_check_fit_input(X,a,r)
        self._oracles=_BayesianOneVsRest(X,a,r,
                                         self.nchoices,
                                         self.beta_prior[1],self.beta_prior[0][0],self.beta_prior[0][1],
                                         self.method)
        return self
    
    def decision_function(self, X):
        """
        Get the scores for each arm following this policy's action-choosing criteria.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to obtain decision function scores for each arm.
        
        Returns
        -------
        scores : array (n_samples, n_choices)
            Scores following this policy for each arm.
        """
        X=_check_X_input(X)
        return self._oracles.predict_ucb(X,self.percentile)
    
    def predict(self, X, exploit=False):
        """
        Selects actions according to this policy for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action according to this policy.
        exploit : bool
            Whether to make a prediction according to the policy, or to just choose the
            arm with the highest expected reward according to current models.
            
        Returns
        -------
        pred : array (n_samples,)
            Actions chosen by the policy.
        """
        if exploit:
            X=_check_X_input(X)
            return np.argmax(self._oracles.predict_avg(X), axis=1)
        return np.argmax(self.decision_function(X), axis=1)


class BayesianTS:
    """
    Bayesian Thompson Sampling
    
    Performs Thompson Sampling by sampling a set of Logistic Regression coefficients
    from each class, then predicting the class with highest estimate.

    Note
    ----
    The implementation here uses PyMC3's GLM formula with default parameters and ADVI.
    This is a very, very slow implementation, and will probably take at least two
    orders or magnitude more to fit compared to other methods.
    
    Parameters
    ----------
    nchoices : int
        Number of arms/labels to choose from.
    method : str, either 'advi' or 'nuts'
        Method used to sample coefficients (see PyMC3's documentation for mode details).
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive samples from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices, 4), 3)
        
    References
    ----------
    [1] An empirical evaluation of thompson sampling (2011)
    """
    def __init__(self, nchoices, method='advi', beta_prior='auto'):

        ## NOTE: this is a really slow and poorly thought implementation
        ## TODO: rewrite using some faster framework such as Edward,
        ##       or with a hard-coded coordinate ascent procedure instead. 
        self.beta_prior = _check_beta_prior(beta_prior, nchoices, 3)
        _check_constructor_input(_ZeroPredictor(), nchoices)
        self.nchoices = nchoices
        assert method in ['advi', 'nuts']
        self.method = method
    
    def fit(self, X, a, r):
        """
        Samples coefficients for Logistic Regression models from partially-labeled data, with
        actions chosen by this same policy.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        a : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            Copy of this same object
        """
        X,a,r=_check_fit_input(X,a,r)
        self._oracles=_BayesianOneVsRest(X,a,r,
                                         self.nchoices,
                                         self.beta_prior[1],self.beta_prior[0][0],self.beta_prior[0][1],
                                         self.method)
        return self
    
    def decision_function(self, X):
        """
        Get the scores for each arm following this policy's action-choosing criteria.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to obtain decision function scores for each arm.
        
        Returns
        -------
        scores : array (n_samples, n_choices)
            Scores following this policy for each arm.
        """
        X=_check_X_input(X)
        return self._oracles.predict_rnd(X)
    
    def predict(self, X, exploit=False):
        """
        Selects actions according to this policy for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action according to this policy.
        exploit : bool
            Whether to make a prediction according to the policy, or to just choose the
            arm with the highest expected reward according to current models.
            
        Returns
        -------
        pred : array (n_samples,)
            Actions chosen by the policy.
        """
        if exploit:
            X=_check_X_input(X)
            return np.argmax(self._oracles.predict_avg(X), axis=1)
        return np.argmax(self.decision_function(X), axis=1)
