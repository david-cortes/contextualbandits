from contextualbandits.utils import _check_constructor_input, _check_beta_prior, _check_fit_input, _check_X_input,\
            _check_1d_inp, _BetaPredictor, _ZeroPredictor, _OnePredictor, _ArrBSClassif, _OneVsRest,\
            _calculate_beta_prior, _BayesianOneVsRest, _BayesianLogisticRegression,\
            _check_bools, _LinUCBSingle, _modify_predict_method
import warnings
import pandas as pd, numpy as np

class BootstrappedUCB:
    """
    Bootstrapped Upper-Confidence Bound

    Obtains an upper confidence bound by taking the percentile of the predictions from a
    set of classifiers, all fit with different bootstrapped samples (multiple samples per arm).
    
    Note
    ----
    When fitting the algorithm to data in batches (online), it's not possible to take an
    exact bootstrapped sample, as the sample is not known in advance. In theory, as the sample size
    grows to infinity, the number of times that an observation appears in a bootstrapped sample is
    distributed ~ Poisson(1). However, I've found that assigning random weights to observations
    produces a more stable effect, so it also has the option to assign weights randomly ~ Gamma(2,2).
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
    nchoices : int
        Number of arms/labels to choose from.
    nsamples : int
        Number of bootstrapped samples per class to take.
    percentile : int [0,100]
        Percentile of the predictions sample to take
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive sampless from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((4/nchoices,4),2)
        Not supported when using 'partial_fit' methods (see alternative 'smooth_predictions').
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    smooth_predictions : bool
        Whether to smooth the predictions for each arm by predicting as:
        yhat_smooth = (yhat*n + 1)/(n + 2)
        where 'n' is the number of times each arm was chosen in the training data
        (not recommended if passing 'beta_prior').
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    batch_sample_method : str, either 'gamma' or 'poisson'
        How to simulate bootstrapped samples when training in batch mode (online).
        See Note.
        """
    def __init__(self, base_algorithm, nchoices, nsamples=10, percentile=80,
                 beta_prior='auto', batch_train=False, smooth_predictions=False,
                 assume_unique_reward=False, batch_sample_method='gamma'):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        assert isinstance(nsamples, int)
        assert nsamples>=2
        assert isinstance(percentile,int) or isinstance(percentile,float)
        assert (percentile>0) and (percentile<100)
        assert (batch_sample_method=='gamma') or (batch_sample_method=='poisson')
        
        self.base_algorithm = base_algorithm
        if beta_prior=='auto':
            self.beta_prior = ((4/nchoices,4),2)
        else:
            self.beta_prior = _check_beta_prior(beta_prior, nchoices, 2)
        self.nchoices = nchoices
        self.nsamples = nsamples
        self.percentile = percentile
        
        batch_train,smooth_predictions,assume_unique_reward=_check_bools(batch_train,
                                        smooth_predictions,assume_unique_reward)
        self.batch_train = batch_train
        self.smooth_predictions = smooth_predictions
        self.assume_unique_reward = assume_unique_reward
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
                                   self.smooth_predictions,
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
    
    def predict(self, X, exploit=False, output_score=False, apply_sigmoid_scores=True):
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
        apply_sigmoid_scores : bool
            If passing output_score=True, whether to apply a sigmoid function to the scores
            from the decision function of the classifier that predicts each class.
            
        Returns
        -------
        pred : array (n_samples,) or (n_samples, 2)
            Actions chosen by the policy. If passing output_score=True, it will be an array
            with the first column indicating the action and the second one indicating the score
            that the classifier gave to that class.
        """
        if exploit:
            X=_check_X_input(X)
            pred=self._oracles.score_avg(X)
        else:
            pred=self.decision_function(X)
            
        if not output_score:
            return np.argmax(pred, axis=1).reshape(-1)
        else:
            score_max=pred.max(axis=1).reshape(-1,1)
            if apply_sigmoid:
                score_max=1/(1+np.exp(-score_max))
            pred=np.argmax(pred, axis=1).reshape(-1,1)
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
    produces a more stable effect, so it also has the option to assign weights randomly ~ Gamma(2,2).
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
    nchoices : int
        Number of arms/labels to choose from.
    nsamples : int
        Number of bootstrapped samples per class to take.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive sampless from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices,4), 1)
        Not supported when using 'partial_fit' methods (see alternative 'smooth_predictions').
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    smooth_predictions : bool
        Whether to smooth the predictions for each arm by predicting as:
        yhat_smooth = (yhat*n + 1)/(n + 2)
        where 'n' is the number of times each arm was chosen in the training data
        (not recommended if passing 'beta_prior').
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
    def __init__(self, base_algorithm, nchoices, nsamples=10, beta_prior='auto', batch_train=False,
                 smooth_predictions=False, assume_unique_reward=False, batch_sample_method='gamma'):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        assert isinstance(nsamples, int)
        assert nsamples>=2
        assert (batch_sample_method=='gamma') or (batch_sample_method=='poisson')
        
        self.beta_prior = _check_beta_prior(beta_prior,nchoices,2)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        self.nsamples=nsamples
        
        batch_train,smooth_predictions,assume_unique_reward=_check_bools(batch_train,
                                        smooth_predictions,assume_unique_reward)
        self.batch_train = batch_train
        self.smooth_predictions = smooth_predictions
        self.assume_unique_reward = assume_unique_reward
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
                                   self.smooth_predictions,
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
    
    def predict(self, X, exploit=False, output_score=False, apply_sigmoid_scores=True):
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
        apply_sigmoid_scores : bool
            If passing output_score=True, whether to apply a sigmoid function to the scores
            from the decision function of the classifier that predicts each class.
            
        Returns
        -------
        pred : array (n_samples,) or (n_samples, 2)
            Actions chosen by the policy. If passing output_score=True, it will be an array
            with the first column indicating the action and the second one indicating the score
            that the classifier gave to that class.
        """
        if exploit:
            X=_check_X_input(X)
            pred=self._oracles.score_avg(X)
        else:
            pred=self.decision_function(X)
        
        if not output_score:
            return np.argmax(pred, axis=1).reshape(-1)
        else:
            score_max=pred.max(axis=1).reshape(-1,1)
            if apply_sigmoid:
                score_max=1/(1+np.exp(-score_max))
            pred=np.argmax(pred, axis=1).reshape(-1,1)
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
    nchoices : int
        Number of arms/labels to choose from.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive sampless from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices,4), 1)
        Not supported when using 'partial_fit' methods (see alternative 'smooth_predictions').
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    smooth_predictions : bool
        Whether to smooth the predictions for each arm by predicting as:
        yhat_smooth = (yhat*n + 1)/(n + 2)
        where 'n' is the number of times each arm was chosen in the training data
        (not recommended if passing 'beta_prior').
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    """
    def __init__(self, base_algorithm, nchoices, beta_prior=None, batch_train=False,
                 smooth_predictions=False, assume_unique_reward=False):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        self.beta_prior = _check_beta_prior(beta_prior,nchoices,1)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        
        batch_train,smooth_predictions,assume_unique_reward=_check_bools(batch_train,
                                        smooth_predictions,assume_unique_reward)
        self.batch_train = batch_train
        self.smooth_predictions = smooth_predictions
        self.assume_unique_reward = assume_unique_reward
    
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
                                   self.smooth_predictions,
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
    
    def predict(self, X, exploit=False, output_score=False, apply_sigmoid_scores=True):
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
        apply_sigmoid_scores : bool
            If passing output_score=True, whether to apply a sigmoid function to the scores
            from the decision function of the classifier that predicts each class.
            
        Returns
        -------
        pred : array (n_samples,) or (n_samples, 2)
            Actions chosen by the policy. If passing output_score=True, it will be an array
            with the first column indicating the action and the second one indicating the score
            that the classifier gave to that class.
        """
        scores=self.decision_function(X)
        pred=np.argmax(scores, axis=1)
        if not output_score:
            return pred
        else:
            score_max=np.max(scores, axis=1).reshape(-1,1)
            if apply_sigmoid_scores:
                score_max=1/(1+np.exp(-score_max))
            return np.c_[pred.reshape(-1,1),score_max]

class EpsilonGreedy:
    """
    Epsilon Greedy
    
    Takes a random action with probability p, or the action with highest
    estimated reward with probability 1-p.
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
    nchoices : int
        Number of arms/labels to choose from.
    explore_prob : float (0,1)
        Probability of taking a random action at each round.
    decay : float (0,1)
        After each prediction, the explore probability reduces to
        p = p*decay
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive sampless from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices,4), 1)
        Not supported when using 'partial_fit' methods (see alternative 'smooth_predictions').
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    smooth_predictions : bool
        Whether to smooth the predictions for each arm by predicting as:
        yhat_smooth = (yhat*n + 1)/(n + 2)
        where 'n' is the number of times each arm was chosen in the training data
        (not recommended if passing 'beta_prior').
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    
    References
    ----------
    [1] The k-armed dueling bandits problem (2010)
    """
    def __init__(self, base_algorithm, nchoices, explore_prob=0.2, decay=0.9999, beta_prior='auto',
                 batch_train=False, smooth_predictions=False, assume_unique_reward=False):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        self.beta_prior = _check_beta_prior(beta_prior,nchoices,1)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        
        assert (explore_prob>0) and (explore_prob<1)
        if decay is not None:
            assert (decay>0) and (decay<1)
            if decay<=.99:
                warnings.warn("Warning: 'EpsilonGreedy' has a very high decay rate.")
        self.explore_prob = explore_prob
        self.decay = decay
        
        batch_train,smooth_predictions,assume_unique_reward=_check_bools(batch_train,
                                        smooth_predictions,assume_unique_reward)
        self.batch_train = batch_train
        self.smooth_predictions = smooth_predictions
        self.assume_unique_reward = assume_unique_reward
    
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
                                   self.smooth_predictions,
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
    
    def predict(self, X, exploit=False, output_score=False, apply_sigmoid_scores=True):
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
        apply_sigmoid_scores : bool
            If passing output_score=True, whether to apply a sigmoid function to the scores
            from the decision function of the classifier that predicts each class.
            
        Returns
        -------
        pred : array (n_samples,) or (n_samples, 2)
            Actions chosen by the policy. If passing output_score=True, it will be an array
            with the first column indicating the action and the second one indicating the score
            that the classifier gave to that class.
        """
        scores=self.decision_function(X)
        pred=np.argmax(scores, axis=1)
        if not exploit:
            ix_change_rnd=(np.random.random(size=X.shape[0])<=self.explore_prob)
            pred[ix_change_rnd] = np.random.randint(self.nchoices, size=ix_change_rnd.sum())
        if self.decay is not None:
            self.explore_prob*=self.decay**X.shape[0]
        if not output_score:
            return pred
        else:
            score_max=np.max(scores, axis=1).reshape(-1,1)
            if apply_sigmoid_scores:
                score_max=1/(1+np.exp(-score_max))
            score_max[ix_change_rnd]=1/self.nchoices
            return np.c_[pred.reshape(-1,1),score_max]


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
        If not None, when there are less than 'n' positive sampless from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices,4), 1)
        Not supported when using 'partial_fit' methods (see alternative 'smooth_predictions').
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    smooth_predictions : bool
        Whether to smooth the predictions for each arm by predicting as:
        yhat_smooth = (yhat*n + 1)/(n + 2)
        where 'n' is the number of times each arm was chosen in the training data
        (not recommended if passing 'beta_prior').
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    
    References
    ----------
    [1] Mortal multi-armed bandits (2009)
    
    """
    def __init__(self, base_algorithm, nchoices, window_size=500, percentile=30, decay=0.9998,
                     decay_type='percentile', initial_thr='auto', fixed_thr=False, greedy_choice=None,
                     beta_prior='auto', batch_train=False, smooth_predictions=False, assume_unique_reward=False):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        self.beta_prior = _check_beta_prior(beta_prior,nchoices,1)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        
        assert isinstance(window_size, int)
        assert isinstance(percentile, int)
        if initial_thr=='auto':
            initial_thr=1/(nchoices/1.5)
        assert isinstance(initial_thr, float)
        assert (percentile>0) and (percentile<100)
        assert window_size>0
        assert isinstance(fixed_thr, bool)
        self.window_size = window_size
        self.percentile = percentile
        self.thr = initial_thr
        self.fixed_thr = fixed_thr
        self.window_cnt = 0
        self.window = np.array([])
        self.decay = decay
        assert (decay_type=='threshold') or (decay_type=='percentile')
        self.decay_type = decay_type
        
        batch_train,smooth_predictions,assume_unique_reward=_check_bools(batch_train,
                                        smooth_predictions,assume_unique_reward)
        self.batch_train = batch_train
        self.smooth_predictions = smooth_predictions
        self.assume_unique_reward = assume_unique_reward

        if greedy_choice is not None:
            assert greedy_choice in ['min', 'max', 'weighted']
            if type(base_algorithm).__name__=='LogisticRegression':
                self.reg=base_algorithm.C
            elif type(base_algorithm).__name__=='RidgeClassifier':
                self.reg=base_algorithm.alpha
            elif type(base_algorithm).__name__=='SGDClassifier':
                self.reg=base_algorithm.alpha
            else:
                raise ValueError("'greedy_choice' requires the classifier to be one of 'LogisticRegression', 'SGDClassifier' or 'RidgeClassifier'")
        self.greedy_choice = greedy_choice
    
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
                                   self.smooth_predictions,
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
        
        # fixed threshold, anything below is always random
        if self.fixed_thr:
            pred_proba=self._oracles.predict_proba(X)
            pred_max=pred_proba.max(axis=1)
            pred=np.argmax(pred_proba, axis=1)
            set_random=pred_max<=self.thr
            pred[set_random]=self._choose_greedy(set_random, self.nchoices, X)
            #pred[set_random]=np.random.randint(self.nchoices, size=set_random.sum())
        else:
            diff_window=self.window_size-self.window_cnt
            
            # case 1: number of predictions to make would still fit within current window
            if diff_window>X.shape[0]:
                pred_proba=self._oracles.predict_proba(X)
                pred_max=pred_proba.max(axis=1)
                pred=np.argmax(pred_proba, axis=1)
                set_random=pred_max<=self.thr
                pred[set_random]=self._choose_greedy(set_random, self.nchoices, X)
                #pred[set_random]=np.random.randint(self.nchoices, size=set_random.sum())
                self.window_cnt+=X.shape[0]
                self.window = np.r_[self.window, pred_max]
                
                # apply decay for all observations
                if self.decay is not None:
                    if self.decay_type=='threshold':
                        self.thr*=self.decay**X.shape[0]
                    if self.decay_type=='percentile':
                        self.percentile*=self.decay**X.shape[0]
            # case 2: number of predictions to make would span more than current window
            else:
                n_take_old_thr=self.window_size-self.window_cnt
                
                pred_proba=self._oracles.predict_proba(X[:n_take_old_thr,:])
                pred_max=pred_proba.max(axis=1)
                pred=np.argmax(pred_proba, axis=1)
                set_random=pred_max<=self.thr
                pred[set_random]=self._choose_greedy(set_random, self.nchoices, X)
                #pred[set_random]=np.random.randint(self.nchoices, size=set_random.sum())
                
                pred_all=np.zeros(X.shape[0])
                pred_all[:n_take_old_thr]=pred
                
                # update threshold, update window
                self.window = np.r_[self.window, pred_max]
                self.thr = np.percentile(self.window, self.percentile)
                self.window = np.array([])
                self.window_cnt = 0
                
                # decay threshold only for these ones
                if self.decay is not None:
                    if self.decay_type=='threshold':
                        self.thr*=self.decay**n_take_old_thr
                    if self.decay_type=='percentile':
                        self.percentile*=self.decay**n_take_old_thr
                
                # rest are calculated recursively
                pred_all[n_take_old_thr:]=self.predict(X[n_take_old_thr:,:])
                return pred_all
                
        return pred

    def _choose_greedy(self, set_greedy, nchoices, X):
        if self.greedy_choice is None:
            return np.random.randint(self.nchoices, size=set_greedy.sum())
        else:
            if self.greedy_choice=='min':
                return ActiveExplorer._get_min_gradient(self, X[set_greedy,:])
            if self.greedy_choice=='max':
                return ActiveExplorer._get_max_gradient(self, X[set_greedy,:])
            if self.greedy_choice=='weighted':
                return ActiveExplorer._get_weighted_gradient(self, X[set_greedy,:])
    
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
    nchoices : int
        Number of arms/labels to choose from.
    explore_rounds : int
        Number of rounds to wait before exploitation mode.
        Will switch after making N predictions.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive sampless from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices,4), 1)
        Not supported when using 'partial_fit' methods (see alternative 'smooth_predictions').
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    smooth_predictions : bool
        Whether to smooth the predictions for each arm by predicting as:
        yhat_smooth = (yhat*n + 1)/(n + 2)
        where 'n' is the number of times each arm was chosen in the training data
        (not recommended if passing 'beta_prior').
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    
    References
    ----------
    [1] The k-armed dueling bandits problem (2012)
    """
    def __init__(self, base_algorithm, nchoices, explore_rounds=2500, beta_prior='auto',
                     batch_train=False, smooth_predictions=False, assume_unique_reward=False):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        self.beta_prior = _check_beta_prior(beta_prior,nchoices,1)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        
        assert explore_rounds>0
        assert isinstance(explore_rounds, int)
        self.explore_rounds = explore_rounds
        self.explore_cnt = 0
        
        batch_train,smooth_predictions,assume_unique_reward=_check_bools(batch_train,
                                        smooth_predictions,assume_unique_reward)
        self.batch_train = batch_train
        self.smooth_predictions = smooth_predictions
        self.assume_unique_reward = assume_unique_reward
    
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
                                   self.smooth_predictions,
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
    
    Logistic Regression which selects a proportion of actions according to an
    active learning heuristic based on gradient.
    
    Note
    ----
    Here, for the predictions that are made according to an active learning heuristic
    (these are selected at random, just like in Epsilon-Greedy), the guiding heuristic
    is the gradient that the observation, having either label (either weighted by the estimted
    probability, or taking the maximum or minimum), would produce on each model that
    predicts a class, given the current coefficients for that model.
    
    Parameters
    ----------
    nchoices : int
        Number of arms/labels to choose from.
    C : float
        Inverse of the regularization parameter for Logistic regression.
        For more details see sklearn.linear_model.LogisticRegression.
        If None, defaults to 1.0.
    explore_prob : float (0,1)
        Probability of selecting an action according to active learning criteria.
    decay : float (0,1)
        After each prediction, the probability of selecting an arm according to active
        learning criteria is set to p = p*decay
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive sampless from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices,4), 1)
        Not supported when using 'partial_fit' methods (see alternative 'smooth_predictions').
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    smooth_predictions : bool
        Whether to smooth the predictions for each arm by predicting as:
        yhat_smooth = (yhat*n + 1)/(n + 2)
        where 'n' is the number of times each arm was chosen in the training data
        (not recommended if passing 'beta_prior').
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_seed : None or int
        Random state or seed to pass to the solver.
    """
    def __init__(self, nchoices, C=None, explore_prob=.15, decay=0.9997, beta_prior='auto',
                     batch_train=False, smooth_predictions=False, assume_unique_reward=False, random_seed=None):
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        if C is None:
            if batch_train:
                base_algorithm = SGDClassifier(loss='log', random_state=random_seed)
                self.reg=1.0
            else:
                base_algorithm = LogisticRegression(solver='lbfgs', random_state=random_seed)
                self.reg=1.0
        else:
            if batch_train:
                base_algorithm = SGDClassifier(alpha = 1/C, loss='log', random_state=random_seed)
                self.reg=C
            else:
                base_algorithm = LogisticRegression(C=C, solver='lbfgs', random_state=random_seed)
                self.reg=C
        
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        self.beta_prior = _check_beta_prior(beta_prior,nchoices,1)

        if batch_train:
            base_algorithm = _modify_predict_method(base_algorithm)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        
        assert isinstance(explore_prob, float)
        assert (explore_prob>0) and (explore_prob<1)
        self.explore_prob = explore_prob
        self.decay = decay
        
        batch_train,smooth_predictions,assume_unique_reward=_check_bools(batch_train,
                                        smooth_predictions,assume_unique_reward)
        self.batch_train = batch_train
        self.smooth_predictions = smooth_predictions
        self.assume_unique_reward = assume_unique_reward
    
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
                                   self.smooth_predictions,
                                   self.assume_unique_reward,
                                   self.batch_train)
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
        X=_check_X_input(X)
        
        if exploit:
            return self._oracles.predict(X)
        
        pred=self._oracles.predict(X)
        change_greedy=np.random.random(size=X.shape[0])<=self.explore_prob
        if change_greedy.sum()>0:
            if gradient_calc=='max':
                pred[change_greedy]=self._get_max_gradient(X[change_greedy,:])
            elif gradient_calc=='min':
                pred[change_greedy]=self._get_min_gradient(X[change_greedy,:])
            elif gradient_calc=='weighted':
                pred[change_greedy]=self._get_weighted_gradient(X[change_greedy,:])
            else:
                raise ValueError("'gradient_cal' must be one of 'max' or 'weighted'")
        
        if self.decay is not None:
            self.explore_prob*=self.decay**X.shape[0]
        
        return pred
        
    def _get_max_gradient(self, X):
        if X.shape[0]==0:
            return np.array([])
        gradmax=np.zeros((X.shape[0],self.nchoices))
        for choice in range(self.nchoices):
            try:
                curr_pred=self._oracles.algos[choice].predict_proba(X)[:,1]
            except:
                curr_pred=np.repeat(0.5, X.shape[0])
            try:
                curr_coef=self._oracles.algos[choice].coef_
                curr_int=self._oracles.algos[choice].intercept_
            except:
                curr_coef=np.zeros(X.shape[1])
                curr_int=0.0
            curr_coef=np.r_[np.array(curr_int).reshape(-1,1),curr_coef.reshape(-1,1)]
            grad_if_zero=np.linalg.norm(curr_pred.reshape(-1,1)*X, axis=1)
            grad_if_one=np.linalg.norm((curr_pred.reshape(-1,1)-1)*X, axis=1)
            grad_max=np.c_[grad_if_zero, grad_if_one].max(axis=1)
            grad_max=grad_max**2 - np.linalg.norm((1/self.reg)*curr_coef)**2
            gradmax[:,choice]=grad_max
        return np.argmax(gradmax, axis=1)
    
    def _get_min_gradient(self, X):
        if X.shape[0]==0:
            return np.array([])
        gradmin=np.zeros((X.shape[0],self.nchoices))
        for choice in range(self.nchoices):
            try:
                curr_pred=self._oracles.algos[choice].predict_proba(X)[:,1]
            except:
                curr_pred=np.repeat(0.5, X.shape[0])
            try:
                curr_coef=self._oracles.algos[choice].coef_
                curr_int=self._oracles.algos[choice].intercept_
            except:
                curr_coef=np.zeros(X.shape[1])
                curr_int=0.0
            curr_coef=np.r_[np.array(curr_int).reshape(-1,1),curr_coef.reshape(-1,1)]
            grad_if_zero=np.linalg.norm(curr_pred.reshape(-1,1)*X, axis=1)
            grad_if_one=np.linalg.norm((curr_pred.reshape(-1,1)-1)*X, axis=1)
            grad_min=np.c_[grad_if_zero, grad_if_one].min(axis=1)
            grad_min=grad_min**2 - np.linalg.norm((1/self.reg)*curr_coef)**2
            gradmin[:,choice]=grad_min
        return np.argmax(gradmin, axis=1)
    
    def _get_weighted_gradient(self, X):
        if X.shape[0]==0:
            return np.array([])
        gradweighted=np.zeros((X.shape[0],self.nchoices))
        for choice in range(self.nchoices):
            try:
                curr_pred=self._oracles.algos[choice].predict_proba(X)[:,1]
            except:
                curr_pred=np.repeat(0.5, X.shape[0])
            try:
                curr_coef=self._oracles.algos[choice].coef_
                curr_int=self._oracles.algos[choice].intercept_
            except:
                curr_coef=np.zeros(X.shape[1])
                curr_int=0.0
            curr_coef=np.r_[np.array(curr_int).reshape(-1,1),curr_coef.reshape(-1,1)]
            grad_if_zero=np.linalg.norm(curr_pred.reshape(-1,1)*X, axis=1)
            grad_if_one=np.linalg.norm((curr_pred.reshape(-1,1)-1)*X, axis=1)
            
            grad_weighted=grad_if_one*curr_pred+grad_if_zero*(1-curr_pred)
            grad_weighted=grad_weighted**2 - np.linalg.norm((1/self.reg)*curr_coef)**2
            gradweighted[:,choice]=grad_weighted
        return np.argmax(gradweighted, axis=1)

class SoftmaxExplorer:
    """
    Soft-Max Explorer
    
    Selects an action according to probabilites determined by a softmax transformation
    on the scores from the decision function that predicts each class.
    
    Note
    ----
    If the base algorithm has 'predict_proba', but no 'decision_function', it will
    calculate the 'probabilities' with a simple scaling by sum rather than by a softmax.
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
    nchoices : int
        Number of arms/labels to choose from.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive sampless from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to auto, will be calculated as:
        beta_prior = ((3/nchoices,4), 1)
        Not supported when using 'partial_fit' methods (see alternative 'smooth_predictions').
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (online),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    smooth_predictions : bool
        Whether to smooth the predictions for each arm by predicting as:
        yhat_smooth = (yhat*n + 1)/(n + 2)
        where 'n' is the number of times each arm was chosen in the training data
        (not recommended if passing 'beta_prior').
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to False,
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    """
    def __init__(self, base_algorithm, nchoices, beta_prior='auto', batch_train=False,
                     smooth_predictions=False, assume_unique_reward=False):
        _check_constructor_input(base_algorithm,nchoices,batch_train)
        self.beta_prior = _check_beta_prior(beta_prior,nchoices,1)
        self.base_algorithm = base_algorithm
        self.nchoices = nchoices
        
        batch_train,smooth_predictions,assume_unique_reward=_check_bools(batch_train,
                                        smooth_predictions,assume_unique_reward)
        self.batch_train = batch_train
        self.smooth_predictions = smooth_predictions
        self.assume_unique_reward = assume_unique_reward
    
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
                                   self.smooth_predictions,
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
        Parameter to control the upper-confidence bound (more is higher).
    
    References
    ----------
    [1] A contextual-bandit approach to personalized news article recommendation (2010)
    """
    def __init__(self, nchoices, alpha=1.0):
        _check_constructor_input(_ZeroPredictor(),nchoices)
        self.alpha=alpha
        self.nchoices=nchoices
        self._oracles=[_LinUCBSingle(self.alpha) for n in range(nchoices)]
    
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
    Bayesian Upper-Confidence Bound
    
    Gets an upper-confidence bound by Bayesian Logistic Regression estimates.
    
    Note
    ----
    The implementation here uses PyMC3's GLM formula with default parameters and ADVI.
    You might want to try building a different one yourself from PyMC3 or Edward.
    The method as implemented here is not scalable to high-dimensional or big datasets.
    
    Parameters
    ----------
    nchoices : int
        Number of arms/labels to choose from.
    percentile : int [0,100]
        Percentile of the predictions sample to take
    method : str, either 'advi' or 'nuts'
        Method used to sample coefficients (see PyMC3's documentation for mode details).
    beta_prior : None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive sampless from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'.
    """
    def __init__(self, nchoices, percentile=80, method='advi', nsamples=None, beta_prior=((3,1),3)):
        import pymc3 as pm
        _check_constructor_input(_BetaPredictor(1,1),nchoices,False)
        self.beta_prior = beta_prior
        self.nchoices = nchoices
        assert method in ['advi','nuts']
        self.method=method
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
    You might want to try building a different one yourself from PyMC3 or Edward.
    The method as implemented here is not scalable to high-dimensional or big datasets.
    
    Parameters
    ----------
    nchoices : int
        Number of arms/labels to choose from.
    method : str, either 'advi' or 'nuts'
        Method used to sample coefficients (see PyMC3's documentation for mode details).
    beta_prior : None, or tuple ((a,b), n)
        If not None, when there are less than 'n' positive sampless from a class
        (actions from that arm that resulted in a reward), it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'.
        
    References
    ----------
    [1] An empirical evaluation of thompson sampling (2011)
    """
    def __init__(self, nchoices, method='advi', beta_prior=((1,1),3)):
        _check_constructor_input(_BetaPredictor(1,1),False)
        self.beta_prior = beta_prior
        self.nchoices = nchoices
        assert method in ['advi','nuts']
        self.method=method
    
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
