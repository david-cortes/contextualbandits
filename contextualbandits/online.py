# -*- coding: utf-8 -*-

import numpy as np, warnings, ctypes
from .utils import _check_constructor_input, _check_beta_prior, \
            _check_smoothing, _check_fit_input, _check_X_input, _check_1d_inp, \
            _ZeroPredictor, _OnePredictor, _OneVsRest,\
            _BootstrappedClassifier_w_predict, _BootstrappedClassifier_w_predict_proba, \
            _BootstrappedClassifier_w_decision_function, _check_njobs, \
            _check_bools, _check_refit_buffer, _check_refit_inp, _check_random_state, \
            _check_autograd_supported, _get_logistic_grads_norms, \
            _gen_random_grad_norms, _gen_zero_norms, \
            _apply_softmax, _apply_inverse_sigmoid, \
            _LinUCB_n_TS_single, _LogisticUCB_n_TS_single, \
            _TreeUCB_n_TS_single
from ._cy_utils import _choice_over_rows, topN_byrow, topN_byrow_softmax

__all__ = ["BootstrappedUCB", "BootstrappedTS",
           "LogisticUCB", "LogisticTS",
           "SeparateClassifiers", "EpsilonGreedy", "AdaptiveGreedy",
           "ExploreFirst", "ActiveExplorer", "SoftmaxExplorer",
           "LinUCB", "LinTS", "ParametricTS",
           "PartitionedUCB", "PartitionedTS"]

class _BasePolicy:
    def _add_common_params(self, base_algorithm, beta_prior, smoothing, noise_to_smooth,
            njobs, nchoices, batch_train, refit_buffer, deep_copy_buffer, assume_unique_reward,
            random_state, assign_algo = True, prior_def_ucb = False,
            force_unfit_predict = False):
        
        if isinstance(base_algorithm, np.ndarray) or base_algorithm.__class__.__name__ == "Series":
            base_algorithm = list(base_algorithm)

        self._add_choices(nchoices)
        _check_constructor_input(base_algorithm, self.nchoices, batch_train)
        self.smoothing = _check_smoothing(smoothing, self.nchoices)
        self.noise_to_smooth = bool(noise_to_smooth)
        self.njobs = _check_njobs(njobs)
        self.batch_train, self.assume_unique_reward = _check_bools(batch_train, assume_unique_reward)
        self.beta_prior = _check_beta_prior(beta_prior, self.nchoices, prior_def_ucb)
        self.random_state = _check_random_state(random_state)
        self.force_unfit_predict = bool(force_unfit_predict)

        if assign_algo:
            self.base_algorithm = base_algorithm
            if ("warm_start" in dir(self.base_algorithm)) and (self.base_algorithm.warm_start):
                self.has_warm_start = True
            else:
                self.has_warm_start = False
        else:
            self.has_warm_start = False

        self.refit_buffer = _check_refit_buffer(refit_buffer, self.batch_train)
        self.deep_copy_buffer = bool(deep_copy_buffer)

        ### For compatibility with the active policies
        self._force_fit = self.force_unfit_predict
        self._force_counters = False

        self.is_fitted = False

    def _add_choices(self, nchoices):
        if isinstance(nchoices, int):
            self.nchoices = nchoices
            self.choice_names = None
        elif isinstance(nchoices, list) or nchoices.__class__.__name__ == "Series" or nchoices.__class__.__name__ == "DataFrame":
            self.choice_names = np.array(nchoices).reshape(-1)
            self.nchoices = self.choice_names.shape[0]
            if np.unique(self.choice_names).shape[0] != self.choice_names.shape[0]:
                raise ValueError("Arm/choice names contain duplicates.")
        elif isinstance(nchoices, np.ndarray):
            self.choice_names = nchoices.reshape(-1)
            self.nchoices = self.choice_names.shape[0]
            if np.unique(self.choice_names).shape[0] != self.choice_names.shape[0]:
                raise ValueError("Arm/choice names contain duplicates.")
        else:
            raise ValueError("'nchoices' must be an integer or list with named arms.")

    def _add_bootstrapped_inputs(self, base_algorithm, batch_sample_method,
                                 nsamples, njobs_samples, percentile,
                                 ts_byrow = False, ts_weighted = False):
        assert (batch_sample_method == 'gamma') or (batch_sample_method == 'poisson')
        assert isinstance(nsamples, int)
        assert nsamples >= 1
        self.batch_sample_method = batch_sample_method
        self.nsamples = nsamples
        self.njobs_samples = _check_njobs(njobs_samples)
        if not isinstance(base_algorithm, list):
            self.base_algorithm = self._make_bootstrapped(base_algorithm, percentile,
                                                          ts_byrow, ts_weighted)
        else:
            self.base_algorithm = [ \
                self._make_bootstrapped(alg, percentile, ts_byrow, ts_weighted) \
                for alg in base_algorithm]

    def _make_bootstrapped(self, base_algorithm, percentile,
                           ts_byrow, ts_weighted):
        if "predict_proba" in dir(base_algorithm):
            return _BootstrappedClassifier_w_predict_proba(
                base_algorithm, self.nsamples, percentile,
                self.batch_train, self.batch_sample_method,
                random_state = 1, ### gets changed later
                njobs = self.njobs_samples,
                ts_byrow = ts_byrow,
                ts_weighted = ts_weighted
                )
        elif "decision_function" in dir(base_algorithm):
            return _BootstrappedClassifier_w_decision_function(
                base_algorithm, self.nsamples, percentile,
                self.batch_train, self.batch_sample_method,
                random_state = 1, ### gets changed later
                njobs = self.njobs_samples,
                ts_byrow = ts_byrow,
                ts_weighted = ts_weighted
                )
        else:
            return _BootstrappedClassifier_w_predict(
                base_algorithm, self.nsamples, percentile,
                self.batch_train, self.batch_sample_method,
                random_state = 1, ### gets changed later
                njobs = self.njobs_samples,
                ts_byrow = ts_byrow,
                ts_weighted = ts_weighted
                )

    def _name_arms(self, pred):
        if self.choice_names is None:
            return pred
        else:
            if not np.issubdtype(pred.dtype, np.integer):
                pred = pred.astype(int)
            return self.choice_names[pred]

    def drop_arm(self, arm_name):
        """
        Drop an arm/choice

        Drops (removes/deletes) an arm from the set of available choices to the policy.

        Note
        ----
        The available arms, if named, are stored in attribute 'choice_names'.
        
        Parameters
        ----------
        arm_name : int or object
            Arm to drop. If passing an integer, will drop at that index (starting at zero). Otherwise,
            will drop the arm matching this name (argument must be of the same type as the individual entries
            passed to 'nchoices' in the initialization).

        Returns
        -------
        self : object
            This object
        """
        if not self.is_fitted:
            raise ValueError("Cannot drop arm from unifitted policy.")
        drop_ix = self._get_drop_ix(arm_name)
        self._oracles._drop_arm(drop_ix)
        self._drop_ix(drop_ix)
        self.has_warm_start = False
        return self

    def _get_drop_ix(self, arm_name):
        if isinstance(arm_name, int):
            if arm_name > self.nchoices:
                raise ValueError("Object has only ", str(self.nchoices), " arms.")
            drop_ix = arm_name
        else:
            if self.choice_names is None:
                raise ValueError("If arms are not named, must pass an integer value.")
            for choice in range(self.nchoices):
                if self.choice_names[choice] == arm_name:
                    drop_ix = choice
                    break
            else:
                raise ValueError("No arm named '", str(arm_name), "' - current names are stored in attribute 'choice_names'.")
        return drop_ix

    def _drop_ix(self, drop_ix):
        if self.choice_names is None:
            self.choice_names = np.arange(self.nchoices)
        self.nchoices -= 1
        self.choice_names = np.r_[self.choice_names[:drop_ix], self.choice_names[drop_ix + 1:]]
        if isinstance(self, _ActivePolicy):
            if isinstance(self._get_grad_norms, list):
                self._get_grad_norms[:drop_ix] + self._get_grad_norms[drop_ix + 1:]
            if isinstance(self._rand_grad_norms, list):
                self._rand_grad_norms[:drop_ix] + self._rand_grad_norms[drop_ix + 1:]
        if isinstance(self.smoothing, np.ndarray):
            self.smoothing = np.c_[self.smoothing[:,:drop_ix], self.smoothing[:,drop_ix + 1:]]

    ## TODO: maybe add functionality to take an arm from another object of this class

    def add_arm(self, arm_name = None, fitted_classifier = None,
                n_w_rew = 0, n_wo_rew = 0, smoothing = None,
                refit_buffer_X = None, refit_buffer_r = None,
                f_grad_norm = None, case_one_class = None):
        """
        Adds a new arm to the pool of choices

        Parameters
        ----------
        arm_name : object
            Name for this arm. Only applicable when using named arms. If None, will use the name of the last
            arm plus 1 (will only work when the names are integers).
        fitted_classifier : object
            If a classifier has already been fit to rewards coming from this arm, you can pass it here, otherwise,
            will be started from the same 'base_classifier' as the initial arms. If using bootstrapped methods or methods from this module which do not
            accept arbitrary classifiers as input,
            don't pass a classifier here (unless using the classes like e.g. `utils._BootstrappedClassifierBase`)
        n_w_rew : int
            Number of trials/rounds with rewards coming from this arm (only used when using a beta prior or smoothing).
        n_wo_rew : int
            Number of trials/rounds without rewards coming from this arm (only used when using a beta prior or smoothing).
        smoothing : None, tuple (a,b), or list
            Smoothing parameters to use for this arm (see documentation of the class constructor
            for details). If ``None`` and if the ``smoothing`` passed to the constructor didn't have
            separate entries per arm, will use the same ``smoothing`` as was passed in the constructor.
            If no ``smoothing`` was passed to the constructor, the ``smoothing`` here will be ignored.
            Must pass a ``smoothing`` here if the constructor was passed a ``smoothing`` with different entries per arm.
        refit_buffer_X : array(m, n) or None
            Refit buffer of 'X' data to use for the new arm. Ignored when using
            'batch_train=False' or 'refit_buffer=None'.
        refit_buffer_r : array(m,) or None
            Refit buffer of rewards data to use for the new arm. Ignored when using
            'batch_train=False' or 'refit_buffer=None'.
        f_grad_norm : function
            Gradient calculation function to use for this arm. This is only
            for the policies that make choices according to active learning
            criteria, and only for situations in which the policy was passed
            different functions for each arm.
        case_one_class : function
            Gradient workaround function for single-class data. This is only
            for the policies that make choices according to active learning
            criteria, and only for situations in which the policy was passed
            different functions for each arm.

        Returns
        -------
        self : object
            This object
        """
        if not self.is_fitted:
            raise ValueError("Cannot add arm to unfitted policy.")
        assert isinstance(n_w_rew,  int)
        assert isinstance(n_wo_rew, int)
        assert n_w_rew >= 0
        assert n_wo_rew >= 0
        refit_buffer_X, refit_buffer_r = \
            _check_refit_inp(refit_buffer_X, refit_buffer_r, self.refit_buffer)
        arm_name = self._check_new_arm_name(arm_name)
        if isinstance(self, _ActivePolicy):
            if isinstance(self._get_grad_norms, list):
                if not callable(f_grad_norm):
                    raise ValueError("'f_grad_norm' must be a function.")
            if isinstance(self._rand_grad_norms, list):
                if not callable(case_one_class):
                    raise ValueError("'case_one_class' must be a function.")
        smoothing = _check_smoothing(smoothing, 1)

        self._oracles._spawn_arm(fitted_classifier, n_w_rew, n_wo_rew,
                                 refit_buffer_X, refit_buffer_r)
        self._append_arm(arm_name, f_grad_norm, case_one_class)
        self._add_to_smoothing(smoothing)
        return self

    def _check_new_arm_name(self, arm_name):
        if self.choice_names is None and arm_name is not None:
            raise ValueError("Cannot create a named arm when no names were passed to 'nchoices'.")
        if arm_name is None and self.choice_names is not None:
            try:
                arm_name = self.choice_names[-1] + 1
            except:
                raise ValueError("Must provide an arm name when using named arms.")
        return arm_name

    def _append_arm(self, arm_name, f_grad_norm, case_one_class):
        if self.choice_names is not None:
            self.choice_names = np.r_[self.choice_names, np.array(arm_name).reshape(-1)]
        if f_grad_norm is not None:
            self._get_grad_norms.append(f_grad_norm)
        if case_one_class is not None:
            self._rand_grad_norms.append(case_one_class)
        self.nchoices += 1

    def _add_to_smoothing(self, smoothing):
        if self.smoothing is None:
            return None
        if (smoothing is None) and (isinstance(self.smoothing, np.ndarray)):
            raise ValueError("Must pass smoothing parameters for new arm.")
        elif smoothing is not None:
            if isinstance(self.smoothing, tuple):
                self.smoothing = np.repeat(np.array(self.smoothing), self.nchoices).reshape((2,-1))
            self.smoothing = np.c_[self.smoothing, np.array(smoothing).reshape((2,1))]

    def fit(self, X, a, r, warm_start=False):
        """
        Fits the base algorithm (one per class [and per sample if bootstrapped]) to partially labeled data.

        Parameters
        ----------
        X : array(n_samples, n_features) or CSR(n_samples, n_features)
            Matrix of covariates for the available data.
        a : array(n_samples, ), int type
            Arms or actions that were chosen for each observations.
        r : array(n_samples, ), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.
        warm_start : bool
            Whether to use the results of previous calls to 'fit' as a start
            for fitting to the 'X' data passed here. This will only be available
            if the base classifier has a property ``warm_start`` too and that
            property is also set to 'True'. You can double-check that it's
            recognized as such by checking this object's property
            ``has_warm_start``. Passing 'True' when the classifier doesn't
            support warm start despite having the property might slow down
            things.
            Dropping arms will make this functionality unavailable.
            This options is not available for 'BootstrappedUCB',
            nor for 'BootstrappedTS'.

        Returns
        -------
        self : obj
            This object
        """
        X, a, r = _check_fit_input(X, a, r, self.choice_names)
        use_warm = warm_start and self.has_warm_start and self.is_fitted
        self._oracles = _OneVsRest(self.base_algorithm,
                                   X, a, r,
                                   self.nchoices,
                                   self.beta_prior[1], self.beta_prior[0][0], self.beta_prior[0][1],
                                   self.random_state,
                                   self.smoothing, self.noise_to_smooth,
                                   self.assume_unique_reward,
                                   self.batch_train,
                                   refit_buffer = self.refit_buffer,
                                   deep_copy = self.deep_copy_buffer,
                                   force_fit = self._force_fit,
                                   force_counters = self._force_counters,
                                   prev_ovr = self._oracles if self.is_fitted else None,
                                   warm = use_warm,
                                   force_unfit_predict = self.force_unfit_predict,
                                   njobs = self.njobs)
        self.is_fitted = True
        return self
    
    def partial_fit(self, X, a, r):
        """
        Fits the base algorithm (one per class) to partially labeled data in batches.
        
        Note
        ----
        In order to use this method, the base classifier must have a 'partial_fit' method,
        such as 'sklearn.linear_model.SGDClassifier'. This method is not available
        for 'LogisticUCB', nor for 'LogisticTS'.

        Parameters
        ----------
        X : array(n_samples, n_features) or CSR(n_samples, n_features)
            Matrix of covariates for the available data.
        a : array(n_samples, ), int type
            Arms or actions that were chosen for each observations.
        r : array(n_samples, ), {0,1}
            Rewards that were observed for the chosen actions. Must be binary rewards 0/1.

        Returns
        -------
        self : obj
            This object
        """
        if not self.batch_train:
            raise ValueError("Must pass 'batch_train' = 'True' to use '.partial_fit'.")
        if '_oracles' in dir(self):
            X, a, r =_check_fit_input(X, a, r, self.choice_names)
            self._oracles.partial_fit(X, a, r)
            self.is_fitted = True
            return self
        else:
            return self.fit(X, a, r)
    
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
        X = _check_X_input(X)
        if not self.is_fitted:
            warnings.warn("Model object has not been fit to data, predictions will be random.")
            return self.random_state.random(size=(X.shape[0], self.nchoices))
        return self._score_matrix(X)

    def _score_matrix(self, X):
        return self._oracles.decision_function(X)

    def _predict_random_if_unfit(self, X, output_score):
        warnings.warn("Model object has not been fit to data, predictions will be random.")
        X = _check_X_input(X)
        pred = self._name_arms(self.random_state.integers(self.nchoices, size = X.shape[0]))
        if not output_score:
            return pred
        else:
            return {"choice" : pred, "score" : (1.0 / self.nchoices) * np.ones(size = X.shape[0], dtype = "float64")}

    def topN(self, X, n):
        """
        Get top-N ranked actions for each observation

        Note
        ----
        This method will rank choices/arms according to what the policy
        dictates - it is not an exploitation-mode rank, so if e.g. there are
        random choices for some observations, there will be random ranks in here.

        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to rank actions according to this policy.
        n : int
            Number of top-ranked actions to output

        Returns
        -------
        topN : array(n_samples, n)
            The top-ranked actions for each observation
        """
        assert n >= 1
        if isinstance(n, float):
            n = int(n)
        assert isinstance(n, int)
        if n > self.nchoices:
            raise ValueError("'n' cannot be greater than 'nchoices'.")
        X = _check_X_input(X)
        scores = self._score_matrix(X)
        if n == self.nchoices:
            topN = np.argsort(scores, axis=1)
        else:
            topN = topN_byrow(scores, n, self.njobs)
        return self._name_arms(topN)


class _BasePolicyWithExploit(_BasePolicy):
    def _exploit(self, X):
        return self._oracles.exploit(X)

    def predict(self, X, exploit = False, output_score = False):
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
        pred : array (n_samples,) or dict("choice" : array(n_samples,), "score" : array(n_samples,))
            Actions chosen by the policy. If passing output_score=True, it will be a dictionary
            with the chosen arm and the score that the arm got following this policy with the classifiers used.
        """
        if not self.is_fitted:
            return self._predict_random_if_unfit(X, output_score)

        if exploit:
            scores = self._exploit(X)
        else:
            scores = self.decision_function(X)
        pred = self._name_arms(np.argmax(scores, axis = 1))

        if not output_score:
            return pred
        else:
            score_max = np.max(scores, axis=1).reshape((-1, 1))
            return {"choice" : pred, "score" : score_max}

class BootstrappedUCB(_BasePolicyWithExploit):
    """
    Bootstrapped Upper Confidence Bound

    Obtains an upper confidence bound by taking the percentile of the predictions from a
    set of classifiers, all fit with different bootstrapped samples (multiple samples per arm).
    
    Note
    ----
    When fitting the algorithm to data in batches (online), it's not possible to take an
    exact bootstrapped sample, as the sample is not known in advance. In theory, as the sample size
    grows to infinity, the number of times that an observation appears in a bootstrapped sample is
    distributed ~ Poisson(1). However, assigning random gamma-distributed weights to observations
    produces a more stable effect, so it also has the option to assign weights randomly ~ Gamma(1,1).
    
    Parameters
    ----------
    base_algorithm : obj or list
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
        Can also pass a list with a different (or already-fit) classifier for each arm.
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    nsamples : int
        Number of bootstrapped samples per class to take.
    percentile : int [0,100]
        Percentile of the predictions sample to take
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((3/log2(nchoices), 4), 2)
        Note that it will only generate one random number per arm, so the 'a'
        parameter should be higher than for other methods.
        This parameter can have a very large impact in the end results, and it's
        recommended to tune it accordingly - scenarios with low expected reward rates
        should have priors that result in drawing small random numbers, whereas
        scenarios with large expected reward rates should have stronger priors and
        tend towards larger random numbers. Also, the more arms there are, the smaller
        the optimal expected value for these random numbers.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (streaming),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    refit_buffer : int or None
        Number of observations per arm to keep as a reserve for passing to
        'partial_fit'. If passing it, up until the moment there are at least this
        number of observations for a given arm, that arm will keep the observations
        when calling 'fit' and 'partial_fit', and will translate calls to
        'partial_fit' to calls to 'fit' with the new plus stored observations.
        After the reserve number is reached, calls to 'partial_fit' will enlarge
        the data batch with the stored observations, and old stored observations
        will be gradually replaced with the new ones (at random, not on a FIFO
        basis). This technique can greatly enchance the performance when fitting
        the data in batches, but memory consumption can grow quite large.
        If passing sparse CSR matrices as input to 'fit' and 'partial_fit',
        these will be converted to dense once they go into this reserve, and
        then converted back to CSR to augment the new data.
        Calls to 'fit' will override this reserve.
        Ignored when passing 'batch_train=False'.
    deep_copy_buffer : bool
        Whether to make deep copies of the data that is stored in the
        reserve for ``refit_buffer``. If passing 'False', when the reserve is
        not yet full, these will only store shallow copies of the data, which
        is faster but will not let Python's garbage collector free memory
        after deleting the data, and if the original data is overwritten, so will
        this buffer.
        Ignored when not using ``refit_buffer``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    batch_sample_method : str, either 'gamma' or 'poisson'
        How to simulate bootstrapped samples when training in batch mode (online).
        See Note.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs_arms : int or None
        Number of parallel jobs to run (for dividing work across arms). If passing None will set it to 1.
        If passing -1 will set it to the number of CPU cores. Note that if the base algorithm is itself
        parallelized, this might result in a slowdown as both compete for available threads, so don't set
        parallelization in both. The total number of parallel jobs will be njobs_arms * njobs_samples. The parallelization uses shared memory, thus you will only
        see a speed up if your base classifier releases the Python GIL, and will
        otherwise result in slower runs.
    njobs_samples : int or None
        Number of parallel jobs to run (for dividing work across samples within one arm). If passing None
        will set it to 1. If passing -1 will set it to the number of CPU cores. The total number of parallel
        jobs will be njobs_arms * njobs_samples.
        The parallelization uses shared memory, thus you will only
        see a speed up if your base classifier releases the Python GIL, and will
        otherwise result in slower runs.

    References
    ----------
    .. [1] Cortes, David. "Adapting multi-armed bandits policies to contextual bandits scenarios."
           arXiv preprint arXiv:1811.04383 (2018).
    """
    def __init__(self, base_algorithm, nchoices, nsamples=10, percentile=80,
                 beta_prior='auto', smoothing=None, noise_to_smooth=True, batch_train=False,
                 refit_buffer=None, deep_copy_buffer=True,
                 assume_unique_reward=False, batch_sample_method='gamma',
                 random_state=None, njobs_arms=-1, njobs_samples=1):
        assert (percentile > 0) and (percentile < 100)
        assert nsamples >= 2
        self._add_common_params(base_algorithm, beta_prior, smoothing, noise_to_smooth, njobs_arms,
                                nchoices, batch_train, refit_buffer, deep_copy_buffer,
                                assume_unique_reward, random_state,
                                assign_algo = False, prior_def_ucb = True)
        self.percentile = percentile
        self._add_bootstrapped_inputs(base_algorithm, batch_sample_method, nsamples, njobs_samples, self.percentile)

    def reset_percentile(self, percentile=80):
        """
        Set the upper confidence bound percentile to a custom number

        Parameters
        ----------
        percentile : int [0,100]
            Percentile of the confidence interval to take.

        Returns
        -------
        self : obj
            This object
        """
        assert (percentile > 0) and (percentile < 100)
        if self.is_fitted:
            self._oracles.reset_attribute("percentile", percentile)
        self.base_algorithm.percentile = percentile
        return self

class BootstrappedTS(_BasePolicyWithExploit):
    """
    Bootstrapped Thompson Sampling
    
    Performs Thompson Sampling by fitting several models per class on bootstrapped samples,
    then makes predictions by taking one of them at random for each class.
    
    Note
    ----
    When fitting the algorithm to data in batches (online), it's not possible to take an
    exact bootstrapped sample, as the sample is not known in advance. In theory, as the sample size
    grows to infinity, the number of times that an observation appears in a bootstrapped sample is
    distributed ~ Poisson(1). However, assigning random gamma-distributed weights to observations
    produces a more stable effect, so it also has the option to assign weights randomly ~ Gamma(1,1).

    Note
    ----
    If you plan to make only one call to 'predict' between calls to 'fit' and have
    ``sample_unique=False``, you can pass ``nsamples=1`` without losing any precision.
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
        Can also pass a list with a different (or already-fit) classifier for each arm.
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    nsamples : int
        Number of bootstrapped samples per class to take.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((2/log2(nchoices), 4), 2)
        This parameter can have a very large impact in the end results, and it's
        recommended to tune it accordingly - scenarios with low expected reward rates
        should have priors that result in drawing small random numbers, whereas
        scenarios with large expected reward rates should have stronger priors and
        tend towards larger random numbers. Also, the more arms there are, the smaller
        the optimal expected value for these random numbers.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    sample_unique : bool
        Whether to use a different bootstrapped classifier per row at each arm when
        calling 'predict'. If passing 'False', will take the same bootstrapped
        classifier within an arm for all the rows passed in a single call to 'predict'.
        Passing 'False' is a faster alternative, but the theoretically correct way
        is using a different one per row.
        Forced to 'True' when passing ``sample_weighted=True``.
    sample_weighted : bool
        Whether to take a weighted average from the predictions from each bootstrapped
        classifier at a given arm, with random weights. This will make the predictions
        more variable (i.e. more randomness in exploration). The alternative (and
        default) is to take a prediction from a single classifier each time.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (streaming),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    refit_buffer : int or None
        Number of observations per arm to keep as a reserve for passing to
        'partial_fit'. If passing it, up until the moment there are at least this
        number of observations for a given arm, that arm will keep the observations
        when calling 'fit' and 'partial_fit', and will translate calls to
        'partial_fit' to calls to 'fit' with the new plus stored observations.
        After the reserve number is reached, calls to 'partial_fit' will enlarge
        the data batch with the stored observations, and old stored observations
        will be gradually replaced with the new ones (at random, not on a FIFO
        basis). This technique can greatly enchance the performance when fitting
        the data in batches, but memory consumption can grow quite large.
        If passing sparse CSR matrices as input to 'fit' and 'partial_fit',
        these will be converted to dense once they go into this reserve, and
        then converted back to CSR to augment the new data.
        Calls to 'fit' will override this reserve.
        Ignored when passing 'batch_train=False'.
    deep_copy_buffer : bool
        Whether to make deep copies of the data that is stored in the
        reserve for ``refit_buffer``. If passing 'False', when the reserve is
        not yet full, these will only store shallow copies of the data, which
        is faster but will not let Python's garbage collector free memory
        after deleting the data, and if the original data is overwritten, so will
        this buffer.
        Ignored when not using ``refit_buffer``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    batch_sample_method : str, either 'gamma' or 'poisson'
        How to simulate bootstrapped samples when training in batch mode (online).
        See Note.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs_arms : int or None
        Number of parallel jobs to run (for dividing work across arms). If passing None will set it to 1.
        If passing -1 will set it to the number of CPU cores. Note that if the base algorithm is itself
        parallelized, this might result in a slowdown as both compete for available threads, so don't set
        parallelization in both. The total number of parallel jobs will be njobs_arms * njobs_samples.
        The parallelization uses shared memory, thus you will only
        see a speed up if your base classifier releases the Python GIL, and will
        otherwise result in slower runs.
    njobs_samples : int or None
        Number of parallel jobs to run (for dividing work across samples within one arm). If passing None
        will set it to 1. If passing -1 will set it to the number of CPU cores. The total number of parallel
        jobs will be njobs_arms * njobs_samples.
        The parallelization uses shared memory, thus you will only
        see a speed up if your base classifier releases the Python GIL, and will
        otherwise result in slower runs.
    
    References
    ----------
    .. [1] Cortes, David. "Adapting multi-armed bandits policies to contextual bandits scenarios."
           arXiv preprint arXiv:1811.04383 (2018).
    .. [2] Chapelle, Olivier, and Lihong Li. "An empirical evaluation of thompson sampling."
           Advances in neural information processing systems. 2011.
    """
    def __init__(self, base_algorithm, nchoices, nsamples=10, beta_prior='auto',
                 smoothing=None, noise_to_smooth=True,
                 sample_unique = True, sample_weighted = False,
                 batch_train=False, refit_buffer=None, deep_copy_buffer=True,
                 assume_unique_reward=False, batch_sample_method='gamma',
                 random_state=None, njobs_arms=-1, njobs_samples=1):
        if sample_weighted:
            sample_unique = True
        self._add_common_params(base_algorithm, beta_prior, smoothing, noise_to_smooth, njobs_arms,
                                nchoices, batch_train, refit_buffer, deep_copy_buffer,
                                assume_unique_reward, random_state, assign_algo=False)
        self._add_bootstrapped_inputs(base_algorithm, batch_sample_method,
                                      nsamples, njobs_samples, None,
                                      ts_byrow = sample_unique, ts_weighted = sample_weighted)

class LogisticUCB(_BasePolicyWithExploit):
    """
    Logistic Regression with Confidence Interval

    Logistic regression classifier which constructs an upper bound on the
    predicted probabilities through a confidence interval calculated from
    the variance-covariance matrix of the predictors.

    Note
    ----
    This strategy is implemented for comparison purposes only and it's not
    recommended to rely on it, particularly not for large datasets.

    Note
    ----
    This strategy does not support fitting the data in batches ('partial_fit'
    will not be available), nor does it support using any other classifier.
    See 'BootstrappedUCB' for a more generalizable version.

    Note
    ----
    This strategy requires each fitted classifier to store a square matrix with
    dimension equal to the number of features. Thus, memory consumption can grow
    very high with this method.

    Parameters
    ----------
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    percentile : int [0,100]
        Percentile of the confidence interval to take.
    fit_intercept : bool
        Whether to add an intercept term to the models.
    lambda_ : float
        Strenght of the L2 regularization. Must be greater than zero.
    ucb_from_empty : bool
        Whether to make upper confidence bounds on arms with no observations according
        to the formula (ties are broken at random for
        them). Choosing this option leads to policies that usually start making random
        predictions until having sampled from all arms, and as such, it's not
        recommended when the number of arms is large relative to the number of rounds.
        Instead, it's recommended to use ``beta_prior``, which acts in the same way
        as for the other policies in this library.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((3/log2(nchoices), 4), 2)
        This parameter can have a very large impact in the end results, and it's
        recommended to tune it accordingly - scenarios with low expected reward rates
        should have priors that result in drawing small random numbers, whereas
        scenarios with large expected reward rates should have stronger priors and
        tend towards larger random numbers. Also, the more arms there are, the smaller
        the optimal expected value for these random numbers.
        Note that this method calculates upper bounds rather than expectations, so the 'a'
        parameter should be higher than for other methods.
        Recommended to use only one of ``beta_prior`` or ``smoothing``. Ignored when
        passing ``ucb_from_empty=True``.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Be aware that the algorithm will use BLAS function calls,
        and if these have multi-threading enabled, it might result in a slow-down
        as both functions compete for available threads.

    References
    ----------
    .. [1] Cortes, David. "Adapting multi-armed bandits policies to contextual bandits scenarios."
           arXiv preprint arXiv:1811.04383 (2018).
    """
    def __init__(self, nchoices, percentile=80, fit_intercept=True,
                 lambda_=1.0, ucb_from_empty=False,
                 beta_prior='auto', smoothing=None, noise_to_smooth=True,
                 assume_unique_reward=False,
                 random_state=None, njobs=-1):
        assert (percentile > 0) and (percentile < 100)
        assert lambda_ > 0.
        base = _LogisticUCB_n_TS_single(lambda_=float(lambda_),
                                        fit_intercept=fit_intercept,
                                        alpha=float(percentile),
                                        ts=False)
        self._add_common_params(base, beta_prior, smoothing, noise_to_smooth, njobs, nchoices,
                                False, None, False, assume_unique_reward,
                                random_state, assign_algo=True, prior_def_ucb=True,
                                force_unfit_predict = ucb_from_empty)
        self.percentile = percentile

    def reset_percentile(self, percentile=80):
        """
        Set the upper confidence bound percentile to a custom number

        Parameters
        ----------
        percentile : int [0,100]
            Percentile of the confidence interval to take.

        Returns
        -------
        self : obj
            This object
        """
        assert (percentile > 0) and (percentile < 100)
        if self.is_fitted:
            self._oracles.reset_attribute("alpha", percentile)
        self.base_algorithm.alpha = percentile
        return self

class LogisticTS(_BasePolicyWithExploit):
    """
    Logistic Regression with Thompson Sampling

    Logistic regression classifier which samples its coefficients using
    the variance-covariance matrix of the predictors, or which samples
    predicted values from a confidence interval as a faster alternative.

    Note
    ----
    This strategy is implemented for comparison purposes only and it's not
    recommended to rely on it, particularly not for large datasets.

    Note
    ----
    This strategy does not support fitting the data in batches ('partial_fit'
    will not be available), nor does it support using any other classifier.
    See 'BootstrappedTS' for a more generalizable version.

    Note
    ----
    This strategy requires each fitted model to store a square matrix with
    dimension equal to the number of features. Thus, memory consumption can grow
    very high with this method.

    Note
    ----
    Be aware that sampling coefficients is an operation that scales poorly with
    the number of columns/features/variables. For wide datasets, it might be
    slower than a bootstrapped approach, especially when using ``sample_unique=True``.

    Parameters
    ----------
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    sample_from : str, one of "coef", "ci"
        Whether to make predictions by sampling the model coefficients or by
        sampling the predicted value from a confidence interval around the best-fit
        coefficients.
    ci_from_empty : bool
        Whether to construct a confidence interval on arms with no observations
        according to a variance-covariance matrix given by the regulatization
        parameter alone.
        Ignored when passing ``sample_from='coef'``.
    multiplier : float
        Multiplier for the covariance matrix. Pass 1 to take it as-is.
        Ignored when passing ``sample_from='ci'``.
    fit_intercept : bool
        Whether to add an intercept term to the models.
    lambda_ : float
        Strenght of the L2 regularization. Must be greater than zero.
    sample_unique : bool
        Whether to sample different coefficients each time a prediction is to
        be made. If passing 'False', when calling 'predict', it will sample
        the same coefficients for all the observations in the same call to
        'predict', whereas if passing 'True', will use a different set of
        coefficients for each observations. Passing 'False' leads to an
        approach which is theoretically wrong, but as sampling coefficients
        can be very slow, using 'False' can provide a reasonable speed up
        without much of a performance penalty.
        Ignored when passing ``sample_from='ci'``.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((2/log2(nchoices), 4), 2)
        This parameter can have a very large impact in the end results, and it's
        recommended to tune it accordingly - scenarios with low expected reward rates
        should have priors that result in drawing small random numbers, whereas
        scenarios with large expected reward rates should have stronger priors and
        tend towards larger random numbers. Also, the more arms there are, the smaller
        the optimal expected value for these random numbers.
        Recommended to use only one of ``beta_prior``, ``smoothing``, ``ci_from_empty``.
        Ignored when passing ``ci_from_empty=True``.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        Recommended to use only one of ``beta_prior``, ``smoothing``, ``ci_from_empty``.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Be aware that the algorithm will use BLAS function calls,
        and if these have multi-threading enabled, it might result in a slow-down
        as both functions compete for available threads.

    References
    ----------
    .. [1] Cortes, David. "Adapting multi-armed bandits policies to contextual bandits scenarios."
           arXiv preprint arXiv:1811.04383 (2018).
    """
    def __init__(self, nchoices, sample_from="ci", ci_from_empty=False, multiplier=1.0,
                 fit_intercept=True, lambda_=1.0, sample_unique=False,
                 beta_prior='auto', smoothing=None, noise_to_smooth=True,
                 assume_unique_reward=False, random_state=None, njobs=-1):
        warnings.warn("This class is experimental. Not recommended to rely on it.")
        assert sample_from in ["ci", "coef"]
        self.sample_from = sample_from
        assert lambda_ > 0.
        assert multiplier > 0.
        base = _LogisticUCB_n_TS_single(lambda_=lambda_,
                                        fit_intercept=fit_intercept,
                                        alpha=0.,
                                        m=multiplier,
                                        ts=True,
                                        ts_from_ci = (sample_from == "ci"),
                                        sample_unique=sample_unique)
        self._add_common_params(base, beta_prior, smoothing, noise_to_smooth, njobs, nchoices,
                                False, None, False, assume_unique_reward,
                                random_state, assign_algo=True, prior_def_ucb=False,
                                force_unfit_predict=ci_from_empty and sample_from == "ci")

class SeparateClassifiers(_BasePolicy):
    """
    Separate Clasifiers per arm
    
    Fits one classifier per arm using only the data on which that arm was chosen.
    Predicts as One-Vs-Rest, plus the usual metaheuristics from ``beta_prior``
    and ``smoothing``.
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
        Can also pass a list with a different (or already-fit) classifier for each arm.
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((2/log2(nchoices), 4), 2)
        This parameter can have a very large impact in the end results, and it's
        recommended to tune it accordingly - scenarios with low expected reward rates
        should have priors that result in drawing small random numbers, whereas
        scenarios with large expected reward rates should have stronger priors and
        tend towards larger random numbers. Also, the more arms there are, the smaller
        the optimal expected value for these random numbers.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (streaming),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    refit_buffer : int or None
        Number of observations per arm to keep as a reserve for passing to
        'partial_fit'. If passing it, up until the moment there are at least this
        number of observations for a given arm, that arm will keep the observations
        when calling 'fit' and 'partial_fit', and will translate calls to
        'partial_fit' to calls to 'fit' with the new plus stored observations.
        After the reserve number is reached, calls to 'partial_fit' will enlarge
        the data batch with the stored observations, and old stored observations
        will be gradually replaced with the new ones (at random, not on a FIFO
        basis). This technique can greatly enchance the performance when fitting
        the data in batches, but memory consumption can grow quite large.
        If passing sparse CSR matrices as input to 'fit' and 'partial_fit',
        these will be converted to dense once they go into this reserve, and
        then converted back to CSR to augment the new data.
        Calls to 'fit' will override this reserve.
        Ignored when passing 'batch_train=False'.
    deep_copy_buffer : bool
        Whether to make deep copies of the data that is stored in the
        reserve for ``refit_buffer``. If passing 'False', when the reserve is
        not yet full, these will only store shallow copies of the data, which
        is faster but will not let Python's garbage collector free memory
        after deleting the data, and if the original data is overwritten, so will
        this buffer.
        Ignored when not using ``refit_buffer``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Note that if the base algorithm is itself parallelized,
        this might result in a slowdown as both compete for available threads, so don't set
        parallelization in both. The parallelization uses shared memory, thus you will only
        see a speed up if your base classifier releases the Python GIL, and will
        otherwise result in slower runs.

    References
    ----------
    .. [1] Cortes, David. "Adapting multi-armed bandits policies to contextual bandits scenarios."
           arXiv preprint arXiv:1811.04383 (2018).
    """
    def __init__(self, base_algorithm, nchoices, beta_prior=None,
                 smoothing=None, noise_to_smooth=True,
                 batch_train=False, refit_buffer=None, deep_copy_buffer=True,
                 assume_unique_reward=False, random_state=None, njobs=-1):
        self._add_common_params(base_algorithm, beta_prior, smoothing, noise_to_smooth, njobs, nchoices,
                                batch_train, refit_buffer, deep_copy_buffer,
                                assume_unique_reward, random_state)
    
    def decision_function_std(self, X):
        """
        Get the predicted "probabilities" from each arm from the classifier that predicts it,
        standardized to sum up to 1 (note that these are no longer probabilities).
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Data for which to obtain decision function scores for each arm.
        
        Returns
        -------
        scores : array (n_samples, n_choices)
            Scores following this policy for each arm.
        """
        X = _check_X_input(X)
        if not self.is_fitted:
            raise ValueError("Object has not been fit to data.")
        return self._oracles.predict_proba(X)
    
    def predict_proba_separate(self, X):
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
        X = _check_X_input(X)
        if not self.is_fitted:
            raise ValueError("Object has not been fit to data.")
        return self._oracles.predict_proba_raw(X)
    
    def predict(self, X, output_score = False):
        """
        Selects actions according to this policy for new data.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to choose an action according to this policy.
        output_score : bool
            Whether to output the score that this method predicted, in case it is desired to use
            it with this pakckage's offpolicy and evaluation modules.
            
        Returns
        -------
        pred : array (n_samples,) or dict("choice" : array(n_samples,), "score" : array(n_samples,))
            Actions chosen by the policy. If passing output_score=True, it will be a dictionary
            with the chosen arm and the score that the arm got following this policy with the classifiers used.
        """
        if not self.is_fitted:
            return self._predict_random_if_unfit(X, output_score)

        scores = self.decision_function(X)
        pred = self._name_arms(np.argmax(scores, axis = 1))

        if not output_score:
            return pred
        else:
            score_max = np.max(scores, axis=1).reshape((-1, 1))
            return {"choice" : pred, "score" : score_max}

class EpsilonGreedy(_BasePolicy):
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
        Can also pass a list with a different (or already-fit) classifier for each arm.
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    explore_prob : float (0,1)
        Probability of taking a random action at each round.
    decay : float (0,1)
        After each prediction, the explore probability reduces to
        p = p*decay
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((2/log2(nchoices), 4), 2)
        The impact of ``beta_prior`` for ``EpsilonGreedy`` is not as high as for other
        policies in this module.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (streaming),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    refit_buffer : int or None
        Number of observations per arm to keep as a reserve for passing to
        'partial_fit'. If passing it, up until the moment there are at least this
        number of observations for a given arm, that arm will keep the observations
        when calling 'fit' and 'partial_fit', and will translate calls to
        'partial_fit' to calls to 'fit' with the new plus stored observations.
        After the reserve number is reached, calls to 'partial_fit' will enlarge
        the data batch with the stored observations, and old stored observations
        will be gradually replaced with the new ones (at random, not on a FIFO
        basis). This technique can greatly enchance the performance when fitting
        the data in batches, but memory consumption can grow quite large.
        If passing sparse CSR matrices as input to 'fit' and 'partial_fit',
        these will be converted to dense once they go into this reserve, and
        then converted back to CSR to augment the new data.
        Calls to 'fit' will override this reserve.
        Ignored when passing 'batch_train=False'.
    deep_copy_buffer : bool
        Whether to make deep copies of the data that is stored in the
        reserve for ``refit_buffer``. If passing 'False', when the reserve is
        not yet full, these will only store shallow copies of the data, which
        is faster but will not let Python's garbage collector free memory
        after deleting the data, and if the original data is overwritten, so will
        this buffer.
        Ignored when not using ``refit_buffer``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Note that if the base algorithm is itself parallelized,
        this might result in a slowdown as both compete for available threads, so don't set
        parallelization in both. The parallelization uses shared memory, thus you will only
        see a speed up if your base classifier releases the Python GIL, and will
        otherwise result in slower runs.
    
    References
    ----------
    .. [1] Cortes, David. "Adapting multi-armed bandits policies to contextual bandits scenarios."
           arXiv preprint arXiv:1811.04383 (2018).
    .. [2] Yue, Yisong, et al. "The k-armed dueling bandits problem."
           Journal of Computer and System Sciences 78.5 (2012): 1538-1556.
    """
    def __init__(self, base_algorithm, nchoices, explore_prob=0.2, decay=0.9999,
                 beta_prior='auto', smoothing=None, noise_to_smooth=True,
                 batch_train=False, refit_buffer=None, deep_copy_buffer=True,
                 assume_unique_reward=False, random_state=None, njobs=-1):
        self._add_common_params(base_algorithm, beta_prior, smoothing, noise_to_smooth, njobs, nchoices,
                                batch_train, refit_buffer, deep_copy_buffer,
                                assume_unique_reward, random_state)
        assert (explore_prob>0) and (explore_prob<1)
        if decay is not None:
            assert (decay>0) and (decay<1)
            if decay <= .99:
                warnings.warn("Warning: 'EpsilonGreedy' has a very high decay rate.")
        self.explore_prob = explore_prob
        self.decay = decay

    def reset_epsilon(self, explore_prob=0.2):
        """
        Set the exploration probability to a custom number

        Parameters
        ----------
        explore_prob : float between 0 and 1
            The exploration probability to set. Note that it will still
            apply the decay after resetting it.

        Returns
        -------
        self : obj
            This object
        """
        assert explore_prob >= 0.
        assert explore_prob <= 1.
        self.explore_prob = explore_prob
        return self
    
    def predict(self, X, exploit = False, output_score = False):
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
        pred : array (n_samples,) or dict("choice" : array(n_samples,), "score" : array(n_samples,))
            Actions chosen by the policy. If passing output_score=True, it will be a dictionary
            with the chosen arm and the score that the arm got following this policy with the classifiers used.
        """
        if not self.is_fitted:
            return self._predict_random_if_unfit(X, output_score)
        scores = self._oracles.decision_function(X)
        pred = np.argmax(scores, axis = 1)
        if not exploit:
            ix_change_rnd = (self.random_state.random(size = X.shape[0]) <= self.explore_prob)
            n_change_rnd = ix_change_rnd.sum()
            pred[ix_change_rnd] = self.random_state.integers(self.nchoices, size = n_change_rnd)
        pred = self._name_arms(pred)

        if self.decay is not None:
            self.explore_prob *= self.decay ** X.shape[0]
        
        if not output_score:
            return pred
        else:
            score_max = np.max(scores, axis = 1).reshape((-1, 1))
            score_max[ix_change_rnd] = 1. / self.nchoices
            return {"choice" : pred, "score" : score_max}

    def _score_matrix(self, X):
        scores = self._oracles.decision_function(X)
        ix_change_rnd = (self.random_state.random(size = X.shape[0]) <= self.explore_prob)
        n_change_rnd = ix_change_rnd.sum()
        scores[ix_change_rnd] = self.random_state.random(size=(n_change_rnd, self.nchoices))

        if self.decay is not None:
            self.explore_prob *= self.decay ** X.shape[0]
        return scores


class _ActivePolicy(_BasePolicy):
    def _check_active_inp(self, base_algorithm, f_grad_norm, case_one_class):
        if f_grad_norm == 'auto':
            if not isinstance(base_algorithm, list):
                _check_autograd_supported(base_algorithm)
            else:
                for alg in base_algorithm:
                    _check_autograd_supported(alg)
            self._get_grad_norms = _get_logistic_grads_norms
        else:
            if not isinstance(f_grad_norm, list):
                assert callable(f_grad_norm)
            else:
                if len(f_grad_norm) != self.nchoices:
                    raise ValueError("'f_grad_norm' must have 'nchoices' entries.")
                for fun in f_grad_norm:
                    if not callable(f_grad_norm):
                        raise ValueError("If passing a list for 'f_grad_norm', " +
                                         "entries must be functions")
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
            if not isinstance(case_one_class, list):
                assert callable(case_one_class)
            else:
                if len(case_one_class) != self.nchoices:
                    raise ValueError("'case_one_class' must have 'nchoices' entries.")
                for fun in case_one_class:
                    if not callable(case_one_class):
                        raise ValueError("If passing a list for 'case_one_class', " +
                                         "entries must be functions")
            self._force_fit = False
            self._rand_grad_norms = case_one_class
        self.case_one_class = case_one_class
        self._force_counters = True

    ### TODO: parallelize this in cython for the default case
    def _crit_active(self, X, pred, grad_crit):
        change_f_grad = isinstance(self._get_grad_norms, list)
        change_r_grad = isinstance(self._rand_grad_norms, list)
        f_grad = self._get_grad_norms
        r_grad = self._rand_grad_norms
        for choice in range(self.nchoices):
            if change_f_grad:
                f_grad = self._get_grad_norms[choice]
            if change_r_grad:
                r_grad = self._rand_grad_norms[choice]

            if self._oracles.should_calculate_grad(choice) or self._force_fit:
                if ( (self._get_grad_norms == _get_logistic_grads_norms)
                      and ("coef_" not in dir(self._oracles.algos[choice]))
                    ):
                    grad_norms = \
                        r_grad(X,
                               self._oracles.get_n_pos(choice),
                               self._oracles.get_n_neg(choice),
                               self._oracles.rng_arm[choice])
                else:
                    grad_norms = f_grad(self._oracles.algos[choice],
                                        X, pred[:, choice])
            else:
                grad_norms = r_grad(X,
                                    self._oracles.get_n_pos(choice),
                                    self._oracles.get_n_neg(choice),
                                    self._oracles.rng_arm[choice])

            if grad_crit == 'min':
                pred[:, choice] = grad_norms.min(axis = 1)
            elif grad_crit == 'max':
                pred[:, choice] = grad_norms.max(axis = 1)
            elif grad_crit == 'weighted':
                pred[:, choice] = np.einsum("i,ij->i", pred[:, choice], grad_norms)
            else:
                raise ValueError("Something went wrong. Please open an issue in GitHub indicating what you were doing.")
        return pred

    def reset_active_choice(self, active_choice='weighted'):
        """
        Set the active gradient criteria to a custom form

        Parameters
        ----------
        active_choice : str in {'min', 'max', 'weighted'}
            How to calculate the gradient that an observation would have on the loss
            function for each classifier, given that it could be either class (positive or negative)
            for the classifier that predicts each arm. If weighted, they are weighted by the same
            probability estimates from the base algorithm.

        Returns
        -------
        self : obj
            This object
        """
        if self.active_choice is None: ### AdaptiveGreedy
            raise ValueError("Cannot change active choice for non-active policy.")
        assert active_choice in ['min', 'max', 'weighted']
        self.active_choice = active_choice
        return self


class AdaptiveGreedy(_ActivePolicy):
    """
    Adaptive Greedy
    
    Takes the action with highest estimated reward, unless that estimation falls below a certain
    threshold, in which case it takes a an action either at random or according to an active learning
    heuristic (same way as `ActiveExplorer`).

    Note
    ----
    The hyperparameters here can make a large impact on the quality of the choices. Be sure
    to tune the threshold (or percentile), decay, and prior (or smoothing parameters).
    
    Note
    ----
    The threshold for the reward probabilities can be set to a hard-coded number, or
    to be calculated dynamically by keeping track of the predictions it makes, and taking
    a fixed percentile of that distribution to be the threshold.
    In the second case, these are calculated in separate batches rather than in a sliding window.
    
    Can also be set to make choices in the same way as
    'ActiveExplorer' rather than random (see 'greedy_choice' parameter).
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
        Can also pass a list with a different (or already-fit) classifier for each arm.
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    window_size : int
        Number of predictions after which the threshold will be updated to the desired percentile.
    percentile : int in [0,100] or None
        Percentile of the predictions sample to set as threshold, below which actions are random.
        If None, will not take percentiles, will instead use the intial threshold and apply decay to it.
    decay : float (0,1) or None
        After each prediction, either the threshold or the percentile gets adjusted to:
            val_t+1 = val_t*decay
    decay_type : str, either 'percentile' or 'threshold'
        Whether to decay the threshold itself or the percentile of the predictions to take after
        each prediction. Ignored when using 'decay=None'. If passing 'percentile=None' and 'decay_type=percentile',
        will be forced to 'threshold'.
    initial_thr : str 'auto' or float (0,1)
        Initial threshold for the prediction below which a random action is taken.
        If set to 'auto', will be calculated as initial_thr = 1 / (2 * sqrt(nchoices)).
        Note that if 'base_algorithm' has a 'decision_function' method, it will first apply a sigmoid function to the
        output, and then compare it to the threshold, so the threshold should lie between zero and one.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((3/nchoices, 4), 2)
        This parameter can have a very large impact in the end results, and it's
        recommended to tune it accordingly - scenarios with low expected reward rates
        should have priors that result in drawing small random numbers, whereas
        scenarios with large expected reward rates should have stronger priors and
        tend towards larger random numbers. Also, the more arms there are, the smaller
        the optimal expected value for these random numbers.
        Note that the default value for ``AdaptiveGreedy`` is different than from the
        other methods in this module, and it's recommended to experiment with different
        values of this hyperparameter.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (streaming),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    refit_buffer : int or None
        Number of observations per arm to keep as a reserve for passing to
        'partial_fit'. If passing it, up until the moment there are at least this
        number of observations for a given arm, that arm will keep the observations
        when calling 'fit' and 'partial_fit', and will translate calls to
        'partial_fit' to calls to 'fit' with the new plus stored observations.
        After the reserve number is reached, calls to 'partial_fit' will enlarge
        the data batch with the stored observations, and old stored observations
        will be gradually replaced with the new ones (at random, not on a FIFO
        basis). This technique can greatly enchance the performance when fitting
        the data in batches, but memory consumption can grow quite large.
        If passing sparse CSR matrices as input to 'fit' and 'partial_fit',
        these will be converted to dense once they go into this reserve, and
        then converted back to CSR to augment the new data.
        Calls to 'fit' will override this reserve.
        Ignored when passing 'batch_train=False'.
    deep_copy_buffer : bool
        Whether to make deep copies of the data that is stored in the
        reserve for ``refit_buffer``. If passing 'False', when the reserve is
        not yet full, these will only store shallow copies of the data, which
        is faster but will not let Python's garbage collector free memory
        after deleting the data, and if the original data is overwritten, so will
        this buffer.
        Ignored when not using ``refit_buffer``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    active_choice : None or str in {'min', 'max', 'weighted'}
        How to select arms when predictions are below the threshold. If passing None, selects them at random (default).
        If passing 'min', 'max' or 'weighted', selects them in the same way as 'ActiveExplorer'.
        Non-random active selection requires being able to calculate gradients (gradients for logistic regression and linear regression (from this package)
        are already defined with an option 'auto' below).
    f_grad_norm : str 'auto', list, or function(base_algorithm, X, pred) -> array (n_samples, 2)
        Function that calculates the row-wise norm of the gradient from observations in X if their class were
        negative (first column) or positive (second column).
        Can also use different functions for each arm, in which case it
        accepts them as a list of functions with length equal to ``nchoices``.
        The option 'auto' will only work with scikit-learn's 'LogisticRegression', 'SGDClassifier', and 'RidgeClassifier';
        with stochQN's 'StochasticLogisticRegression';
        and with this package's 'LinearRegression'.
    case_one_class : str 'auto', 'zero', None, list, or function(X, n_pos, n_neg, rng) -> array(n_samples, 2)
        If some arm/choice/class has only rewards of one type, many models will fail to fit, and consequently the gradients
        will be undefined. Likewise, if the model has not been fit, the gradient might also be undefined, and this requires a workaround.
            * If passing 'None', will assume that ``base_algorithm`` can be fit to
              data of only-positive or only-negative class without problems, and that
              it can calculate gradients and predictions with a ``base_algorithm``
              object that has not been fitted. Be aware that the methods 'predict',
              'predict_proba', and 'decision_function' in ``base_algorithm`` might be
              overwritten with another method that wraps it in a try-catch block, so
              don't rely on it producing errors when unfitted.
            * If passing a function, will take the output of it as the row-wise
              gradient norms when it compares them against other arms/classes, with
              the first column having the values if the observations were of negative
              class, and the second column if they were positive class. The other
              inputs to this function are the number of positive and negative examples
              that have been observed, and a ``Generator`` object from NumPy to use
              for generating random numbers.
            * If passing a list, will assume each entry is a function as described
              above, to be used with each corresponding arm.
            * If passing 'auto', will generate random numbers:

                * negative: ~ Gamma(log10(n_features) / (n_pos+1)/(n_pos+n_neg+2), log10(n_features)).

                * positive: ~ Gamma(log10(n_features) * (n_pos+1)/(n_pos+n_neg+2), log10(n_features)).

            * If passing 'zero', it will output zero whenever models have not been fitted.
        Note that the theoretically correct approach for a logistic regression would
        be to assume models with all-zero coefficients, in which case the gradient
        is defined in the absence of any data, but this tends to produce bad end
        results.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Note that if the base algorithm is itself parallelized,
        this might result in a slowdown as both compete for available threads, so don't set
        parallelization in both. The parallelization uses shared memory, thus you will only
        see a speed up if your base classifier releases the Python GIL, and will
        otherwise result in slower runs.
    
    References
    ----------
    .. [1] Chakrabarti, Deepayan, et al. "Mortal multi-armed bandits."
           Advances in neural information processing systems. 2009.
    .. [2] Cortes, David. "Adapting multi-armed bandits policies to contextual bandits scenarios."
           arXiv preprint arXiv:1811.04383 (2018).
    """
    def __init__(self, base_algorithm, nchoices, window_size=500, percentile=30,
                 decay=0.9998, decay_type='percentile', initial_thr='auto',
                 beta_prior='auto', smoothing=None, noise_to_smooth=True,
                 batch_train=False, refit_buffer=None,  deep_copy_buffer=True,
                 assume_unique_reward=False, active_choice=None, f_grad_norm='auto',
                 case_one_class='auto', random_state=None, njobs=-1):
        if beta_prior == "auto":
            beta_prior = ((3./nchoices, 4.), 2)
        self._add_common_params(base_algorithm, beta_prior, smoothing, noise_to_smooth, njobs, nchoices,
                                batch_train, refit_buffer, deep_copy_buffer,
                                assume_unique_reward, random_state)
        
        assert isinstance(window_size, int)
        if percentile is not None:
            assert isinstance(percentile, int)
            assert (percentile > 0) and (percentile < 100)
        if initial_thr == 'auto':
            if not isinstance(nchoices, list):
                initial_thr = 1.0 / (np.sqrt(nchoices) * 2.0)
            else:
                initial_thr = 1.0 / (np.sqrt(len(nchoices)) * 2.0)
        assert isinstance(initial_thr, float)
        assert window_size > 0
        self.window_size = window_size
        self.percentile = percentile
        self.thr = initial_thr
        self.window_cnt = 0
        self.window = np.array([])
        assert (decay_type == 'threshold') or (decay_type == 'percentile')
        if (decay_type == 'percentile') and (percentile is None):
            decay_type = 'threshold'
        self.decay_type = decay_type
        if decay is not None:
            assert (decay >= 0.0) and (decay <= 1.0)
        if (decay_type == 'percentile') and (percentile is None):
            decay = 1.
        self.decay = decay

        if active_choice is not None:
            assert active_choice in ['min', 'max', 'weighted']
            self._check_active_inp(base_algorithm, f_grad_norm, case_one_class)
        self.active_choice = active_choice

    def reset_threshold(self, threshold="auto"):
        """
        Set the adaptive threshold to a custom number

        Parameters
        ----------
        threshold : float or "auto"
            New threshold to use. If passing "auto", will set it
            to 1.5/nchoices. Note that this threshold will still be
            decayed if the object was initialized with ``decay_type="threshold"``,
            and will still be updated if initialized with ``percentile != None``.

        Returns
        -------
        self : obj
            This object
        """
        if isinstance(threshold, int):
            threshold = float(threshold)
        elif threshold == "auto":
            threshold = 1.5 / self.nchoices
        assert isinstance(threshold, float)
        self.thr = threshold
        return self

    def reset_percentile(self, percentile=30):
        """
        Set the moving percentile to a custom number

        Parameters
        ----------
        percentile : int between 0 and 100
            The new percentile to set. Note that it will still apply
            decay to it after being set through this method.

        Returns
        -------
        self : obj
            This object
        """
        if self.decay_type == 'threshold':
            raise ValueError("Method is not available when not using percentile decay.")
        assert percentile >= 0
        assert percentile <= 100
        self.percentile = percentile
        return self

    def predict(self, X, exploit = False):
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
        if not self.is_fitted:
            return self._predict_random_if_unfit(X, False)
        return self._name_arms(self._predict(X, exploit, True))
    
    def _predict(self, X, exploit = False, choose = True):
        
        if X.shape[0] == 0:
            if choose:
                return np.array([])
            else:
                return np.empty((0, self.nchoices), dtype=ctypes.c_double)
        
        if exploit:
            if choose:
                return self._oracles.predict(X)
            else:
                return self._oracles.decision_function(X)
        
        # fixed threshold, anything below is always random
        if (self.decay == 1) or (self.decay is None):
            pred, pred_max = self._calc_preds(X, choose)

        # variable threshold that needs to be updated
        else:
            remainder_window = self.window_size - self.window_cnt
            
            # case 1: number of predictions to make would still fit within current window
            if remainder_window > X.shape[0]:
                pred, pred_max = self._calc_preds(X, choose)
                self.window_cnt += X.shape[0]
                self.window = np.r_[self.window, pred_max]
                
                # apply decay for all observations
                self._apply_decay(X.shape[0])

            # case 2: number of predictions to make would span more than current window
            else:
                # predict for the remainder of this window
                pred, pred_max = self._calc_preds(X[:remainder_window, :], choose)
                
                # allocate the rest that don't fit in this window
                if choose:
                    pred_all = np.zeros(X.shape[0])
                else:
                    pred_all = np.zeros((X.shape[0], self.nchoices), dtype=ctypes.c_double)
                pred_all[:remainder_window] = pred
                
                # complete window, update percentile if needed
                self.window = np.r_[self.window, pred_max]
                if self.decay_type == 'percentile':
                    self.thr = np.percentile(self.window, self.percentile)

                # reset window
                self.window = np.array([])
                self.window_cnt = 0
                
                # decay threshold only for these observations
                self._apply_decay(remainder_window)
                
                # predict the rest recursively
                pred_all[remainder_window:] = self._predict(X[remainder_window:, :], False, choose)
                return pred_all
                
        return pred

    def _apply_decay(self, nobs):
        if (self.decay is not None) and (self.decay != 1):
            if self.decay_type == 'threshold':
                self.thr *= self.decay ** nobs
            elif self.decay_type == 'percentile':
                self.percentile *= self.decay ** nobs
            else:
                raise ValueError("'decay_type' must be one of 'threshold' or 'percentile'")

    def _calc_preds(self, X, choose = True):
        pred_proba = self._oracles.decision_function(X)
        pred_max = pred_proba.max(axis = 1)
        if choose:
            pred = np.argmax(pred_proba, axis = 1)
        else:
            pred = pred_proba
        set_greedy = pred_max <= self.thr
        if np.any(set_greedy):
            self._choose_greedy(set_greedy, X, pred, pred_proba, choose)
        return pred, pred_max

    def _choose_greedy(self, set_greedy, X, pred, pred_all, choose = True):
        if self.active_choice is None:
            n_greedy = set_greedy.sum()
            if choose:
                pred[set_greedy] = self.random_state.integers(self.nchoices, size = n_greedy)
            else:
                pred[set_greedy] = self.random_state.random(size = (n_greedy, self.nchoices))
        else:
            scores = self._crit_active(
                        X[set_greedy],
                        pred_all[set_greedy],
                        self.active_choice)
            if choose:
                pred[set_greedy] = np.argmax(scores, axis = 1)
            else:
                pred[set_greedy] = scores

    def _score_matrix(self, X):
        return self._predict(X, False, False)

class ExploreFirst(_ActivePolicy):
    """
    Explore First, a.k.a. Explore-Then-Exploit
    
    Selects random actions for the first N predictions, after which it selects the
    best arm only, according to its estimates.
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
        Can also pass a list with a different (or already-fit) classifier for each arm.
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    explore_rounds : int
        Number of rounds to wait before exploitation mode.
        Will switch after making N predictions.
    prob_active_choice : float (0, 1)
        Probability of choosing explore-mode actions according to active
        learning criteria. Pass zero for choosing everything at random.
    active_choice : str, one of 'weighted', 'max' or 'min'
        How to calculate the gradient that an observation would have on the loss
        function for each classifier, given that it could be either class (positive or negative)
        for the classifier that predicts each arm. If weighted, they are weighted by the same
        probability estimates from the base algorithm.
    f_grad_norm : str 'auto' or function(base_algorithm, X, pred) -> array (n_samples, 2)
        Function that calculates the row-wise norm of the gradient from observations in X if their class were
        negative (first column) or positive (second column).
        Can also use different functions for each arm, in which case it
        accepts them as a list of functions with length equal to ``nchoices``.
        The option 'auto' will only work with scikit-learn's 'LogisticRegression', 'SGDClassifier' (log-loss only), and 'RidgeClassifier';
        with stochQN's 'StochasticLogisticRegression';
        and with this package's 'LinearRegression'.
        Ignored when passing ``prob_active_choice=0.``
    case_one_class : str 'auto', 'zero', None, or function(X, n_pos, n_neg, rng) -> array(n_samples, 2)
        If some arm/choice/class has only rewards of one type, many models will fail to fit, and consequently the gradients
        will be undefined. Likewise, if the model has not been fit, the gradient might also be undefined, and this requires a workaround.
            * If passing 'None', will assume that ``base_algorithm`` can be fit to
              data of only-positive or only-negative class without problems, and that
              it can calculate gradients and predictions with a ``base_algorithm``
              object that has not been fitted. Be aware that the methods 'predict',
              'predict_proba', and 'decision_function' in ``base_algorithm`` might be
              overwritten with another method that wraps it in a try-catch block, so
              don't rely on it producing errors when unfitted.
            * If passing a function, will take the output of it as the row-wise
              gradient norms when it compares them against other arms/classes, with
              the first column having the values if the observations were of negative
              class, and the second column if they were positive class. The other
              inputs to this function are the number of positive and negative examples
              that have been observed, and a ``Generator`` object from NumPy to use
              for generating random numbers.
            * If passing a list, will assume each entry is a function as described
              above, to be used with each corresponding arm.
            * If passing 'auto', will generate random numbers:

                * negative: ~ Gamma(log10(n_features) / (n_pos+1)/(n_pos+n_neg+2), log10(n_features)).

                * positive: ~ Gamma(log10(n_features) * (n_pos+1)/(n_pos+n_neg+2), log10(n_features)).

            * If passing 'zero', it will output zero whenever models have not been fitted.
        Note that the theoretically correct approach for a logistic regression would
        be to assume models with all-zero coefficients, in which case the gradient
        is defined in the absence of any data, but this tends to produce bad end
        results.
        Ignored when passing ``prob_active_choice=0.``
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((2/log2(nchoices), 4), 2)
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (streaming),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    refit_buffer : int or None
        Number of observations per arm to keep as a reserve for passing to
        'partial_fit'. If passing it, up until the moment there are at least this
        number of observations for a given arm, that arm will keep the observations
        when calling 'fit' and 'partial_fit', and will translate calls to
        'partial_fit' to calls to 'fit' with the new plus stored observations.
        After the reserve number is reached, calls to 'partial_fit' will enlarge
        the data batch with the stored observations, and old stored observations
        will be gradually replaced with the new ones (at random, not on a FIFO
        basis). This technique can greatly enchance the performance when fitting
        the data in batches, but memory consumption can grow quite large.
        If passing sparse CSR matrices as input to 'fit' and 'partial_fit',
        these will be converted to dense once they go into this reserve, and
        then converted back to CSR to augment the new data.
        Calls to 'fit' will override this reserve.
        Ignored when passing 'batch_train=False'.
    deep_copy_buffer : bool
        Whether to make deep copies of the data that is stored in the
        reserve for ``refit_buffer``. If passing 'False', when the reserve is
        not yet full, these will only store shallow copies of the data, which
        is faster but will not let Python's garbage collector free memory
        after deleting the data, and if the original data is overwritten, so will
        this buffer.
        Ignored when not using ``refit_buffer``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Note that if the base algorithm is itself parallelized,
        this might result in a slowdown as both compete for available threads, so don't set
        parallelization in both. The parallelization uses shared memory, thus you will only
        see a speed up if your base classifier releases the Python GIL, and will
        otherwise result in slower runs.

    References
    ----------
    .. [1] Cortes, David. "Adapting multi-armed bandits policies to contextual bandits scenarios."
           arXiv preprint arXiv:1811.04383 (2018).
    """
    def __init__(self, base_algorithm, nchoices, explore_rounds=2500,
                 prob_active_choice=0., active_choice='weighted',
                 f_grad_norm='auto', case_one_class='auto',
                 beta_prior=None, smoothing=None, noise_to_smooth=True,
                 batch_train=False, refit_buffer=None, deep_copy_buffer=True,
                 assume_unique_reward=False, random_state=None, njobs=-1):
        self._add_common_params(base_algorithm, beta_prior, smoothing, noise_to_smooth, njobs, nchoices,
                                batch_train, refit_buffer, deep_copy_buffer,
                                assume_unique_reward, random_state)
        
        assert explore_rounds>0
        assert isinstance(explore_rounds, int)
        self.explore_rounds = explore_rounds
        self.explore_cnt = 0

        assert (prob_active_choice >= 0.) and (prob_active_choice <= 1.)
        self.prob_active_choice = float(prob_active_choice)
        if self.prob_active_choice > 0:
            assert active_choice in ['min', 'max', 'weighted']
            self.active_choice = active_choice
            self._check_active_inp(base_algorithm, f_grad_norm, case_one_class)
        else:
            self.active_choice = None

    def reset_count(self):
        """
        Resets the counter for exploitation mode

        Returns
        -------
        self

        """
        self.explore_cnt = 0
        return self

    def predict(self, X, exploit = False):
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
        if not self.is_fitted:
            return self._predict_random_if_unfit(X, False)
        return self._name_arms(self._predict(X, exploit))
    
    def _predict(self, X, exploit = False):
        X = _check_X_input(X)
        
        if X.shape[0] == 0:
            return np.array([])
        
        if exploit:
            return self._oracles.predict(X)
        
        if self.explore_cnt < self.explore_rounds:
            self.explore_cnt += X.shape[0]
            
            # case 1: all predictions are within allowance
            if self.explore_cnt <= self.explore_rounds:
                pred = self.random_state.integers(self.nchoices, size = X.shape[0])
                self._choose_active(X, pred)
                return pred
            
            # case 2: some predictions are within allowance, others are not
            else:
                n_explore = self.explore_rounds - self.explore_cnt + X.shape[0]
                pred = np.empty(X.shape[0], type = ctypes.c_double)
                pred[:n_explore] = self.random_state.integers(self.nchoices, n_explore)
                self._choose_active(X[:n_explore], pred[:n_explore])
                pred[n_explore:] = self._oracles.predict(X[n_explore:])
                return pred
        else:
            return self._oracles.predict(X)

    def _score_matrix(self, X):
        if self.explore_cnt < self.explore_rounds:
            self.explore_cnt += X.shape[0]

            # case 1: all predictions are within allowance
            if self.explore_cnt <= self.explore_rounds:
                scores = self.random_state.random(size=(X.shape[0], self.nchoices))
                self._choose_active(X, scores, choose=False)
            
            # case 2: some predictions are within allowance, others are not
            else:
                scores = np.empty((X.shape[0], self.nchoices), type = ctypes.c_double)
                scores[:n_explore] = self.random_state.random(size=(n_explore, self.nchoices))
                self._choose_active(X[:n_explore], scores[:n_explore], choose=False)
                scores[n_explore:] = self._oracles.decision_function(X[n_explore:])
            
        else:
            scores = self._oracles.decision_function(X)

        return scores

    def _choose_active(self, X, pred, choose=True):
        if self.prob_active_choice <= 0.:
            return None

        pick_active = self.random_state.random(size=X.shape[0]) <= self.prob_active_choice
        if not np.any(pick_active):
            return None
        by_crit = self._crit_active(
                        X[pick_active],
                        self._oracles.decision_function(X[pick_active]),
                        self.active_choice)
        if choose:
            pred[pick_active] = np.argmax(by_crit, axis = 1)
        else:
            pred[pick_active] = by_crit


class ActiveExplorer(_ActivePolicy, _BasePolicyWithExploit):
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
    predicts a class, given the current coefficients for that model. This of course requires
    being able to calculate gradients - package comes with pre-defined gradient functions for
    linear and logistic regression, and allows passing custom functions for others.
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
        Can also pass a list with a different (or already-fit) classifier for each arm.
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    f_grad_norm : str 'auto' or function(base_algorithm, X, pred) -> array (n_samples, 2)
        Function that calculates the row-wise norm of the gradient from observations in X if their class were
        negative (first column) or positive (second column).
        Can also use different functions for each arm, in which case it
        accepts them as a list of functions with length equal to ``nchoices``.
        The option 'auto' will only work with scikit-learn's 'LogisticRegression', 'SGDClassifier' (log-loss only), and 'RidgeClassifier';
        with stochQN's 'StochasticLogisticRegression';
        and with this package's 'LinearRegression'.
    case_one_class : str 'auto', 'zero', None, or function(X, n_pos, n_neg, rng) -> array(n_samples, 2)
        If some arm/choice/class has only rewards of one type, many models will fail to fit, and consequently the gradients
        will be undefined. Likewise, if the model has not been fit, the gradient might also be undefined, and this requires a workaround.
            * If passing 'None', will assume that ``base_algorithm`` can be fit to
              data of only-positive or only-negative class without problems, and that
              it can calculate gradients and predictions with a ``base_algorithm``
              object that has not been fitted. Be aware that the methods 'predict',
              'predict_proba', and 'decision_function' in ``base_algorithm`` might be
              overwritten with another method that wraps it in a try-catch block, so
              don't rely on it producing errors when unfitted.
            * If passing a function, will take the output of it as the row-wise
              gradient norms when it compares them against other arms/classes, with
              the first column having the values if the observations were of negative
              class, and the second column if they were positive class. The other
              inputs to this function are the number of positive and negative examples
              that have been observed, and a ``Generator`` object from NumPy to use
              for generating random numbers.
            * If passing a list, will assume each entry is a function as described
              above, to be used with each corresponding arm.
            * If passing 'auto', will generate random numbers:

                * negative: ~ Gamma(log10(n_features) / (n_pos+1)/(n_pos+n_neg+2), log10(n_features)).

                * positive: ~ Gamma(log10(n_features) * (n_pos+1)/(n_pos+n_neg+2), log10(n_features)).

            * If passing 'zero', it will output zero whenever models have not been fitted.
        Note that the theoretically correct approach for a logistic regression would
        be to assume models with all-zero coefficients, in which case the gradient
        is defined in the absence of any data, but this tends to produce bad end
        results.
    active_choice : str in {'min', 'max', 'weighted'}
        How to calculate the gradient that an observation would have on the loss
        function for each classifier, given that it could be either class (positive or negative)
        for the classifier that predicts each arm. If weighted, they are weighted by the same
        probability estimates from the base algorithm.
    explore_prob : float (0,1)
        Probability of selecting an action according to active learning criteria.
    decay : float (0,1)
        After each prediction, the probability of selecting an arm according to active
        learning criteria is set to p = p*decay
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((2/log2(nchoices), 4), 2)
        This parameter can have a very large impact in the end results, and it's
        recommended to tune it accordingly - scenarios with low expected reward rates
        should have priors that result in drawing small random numbers, whereas
        scenarios with large expected reward rates should have stronger priors and
        tend towards larger random numbers. Also, the more arms there are, the smaller
        the optimal expected value for these random numbers.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (streaming),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    refit_buffer : int or None
        Number of observations per arm to keep as a reserve for passing to
        'partial_fit'. If passing it, up until the moment there are at least this
        number of observations for a given arm, that arm will keep the observations
        when calling 'fit' and 'partial_fit', and will translate calls to
        'partial_fit' to calls to 'fit' with the new plus stored observations.
        After the reserve number is reached, calls to 'partial_fit' will enlarge
        the data batch with the stored observations, and old stored observations
        will be gradually replaced with the new ones (at random, not on a FIFO
        basis). This technique can greatly enchance the performance when fitting
        the data in batches, but memory consumption can grow quite large.
        If passing sparse CSR matrices as input to 'fit' and 'partial_fit',
        these will be converted to dense once they go into this reserve, and
        then converted back to CSR to augment the new data.
        Calls to 'fit' will override this reserve.
        Ignored when passing 'batch_train=False'.
    deep_copy_buffer : bool
        Whether to make deep copies of the data that is stored in the
        reserve for ``refit_buffer``. If passing 'False', when the reserve is
        not yet full, these will only store shallow copies of the data, which
        is faster but will not let Python's garbage collector free memory
        after deleting the data, and if the original data is overwritten, so will
        this buffer.
        Ignored when not using ``refit_buffer``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Note that if the base algorithm is itself parallelized,
        this might result in a slowdown as both compete for available threads, so don't set
        parallelization in both. The parallelization uses shared memory, thus you will only
        see a speed up if your base classifier releases the Python GIL, and will
        otherwise result in slower runs.

    References
    ----------
    .. [1] Cortes, David. "Adapting multi-armed bandits policies to contextual bandits scenarios."
           arXiv preprint arXiv:1811.04383 (2018).
    """
    def __init__(self, base_algorithm, nchoices,
                 f_grad_norm='auto', case_one_class='auto', active_choice='weighted',
                 explore_prob=.15, decay=0.9997,
                 beta_prior='auto', smoothing=None, noise_to_smooth=True,
                 batch_train=False, refit_buffer=None, deep_copy_buffer=True,
                 assume_unique_reward=False, random_state=None, njobs=-1):
        assert active_choice in ['min', 'max', 'weighted']
        self.active_choice = active_choice
        self._check_active_inp(base_algorithm, f_grad_norm, case_one_class)
        self._add_common_params(base_algorithm, beta_prior, smoothing, noise_to_smooth, njobs, nchoices,
                                batch_train, refit_buffer, deep_copy_buffer,
                                assume_unique_reward, random_state)
        assert isinstance(explore_prob, float)
        assert (explore_prob > 0.) and (explore_prob <= 1.)
        self.explore_prob = explore_prob
        self.decay = decay

    def reset_explore_prob(self, explore_prob=0.2):
        """
        Set the active exploration probability to a custom number

        Parameters
        ----------
        explore_prob : float between 0 and 1
            The new exploration probability. Note that it will still apply
            decay on it after being reset.

        Returns
        -------
        self : obj
            This object
        """
        assert explore_prob >= 0.
        assert explore_prob <= 1.
        self.explore_prob = explore_prob
        return self

    def _score_matrix(self, X, exploit=False):
        pred = self._oracles.decision_function(X)
        if not exploit:
            change_greedy = self.random_state.random(size=X.shape[0]) <= self.explore_prob
            if np.any(change_greedy):
                pred[change_greedy, :] = self._crit_active(
                                            X[change_greedy, :],
                                            pred[change_greedy, :],
                                            self.active_choice)
            
            if self.decay is not None:
                self.explore_prob *= self.decay ** X.shape[0]
        return pred

    def _exploit(self, X):
        return self._oracles.decision_function(X)

class SoftmaxExplorer(_BasePolicy):
    """
    SoftMax Explorer
    
    Selects an action according to probabilites determined by a softmax transformation
    on the scores from the decision function that predicts each class.

    Note
    ----
    Will apply an inverse sigmoid transformations to the probabilities that come from the base algorithm
    before applying the softmax function.
    
    
    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1, to which it
               will apply an inverse sigmoid function.
            2) A 'decision_function' method with unbounded outputs (n_samples,).
            3) A 'predict' method outputting (n_samples,), values in [0,1], to which it will apply an inverse sigmoid function.
        Can also pass a list with a different (or already-fit) classifier for each arm.
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    multiplier : float or None
        Number by which to multiply the outputs from the base algorithm before applying the softmax function
        (i.e. will take softmax(yhat * multiplier)).
    inflation_rate : float or None
        Number by which to multiply the multipier rate after every prediction, i.e. after making
        't' predictions, the multiplier will be 'multiplier_t = multiplier * inflation_rate^t'.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((2/log2(nchoices), 4), 2)
        This parameter can have a very large impact in the end results, and it's
        recommended to tune it accordingly - scenarios with low expected reward rates
        should have priors that result in drawing small random numbers, whereas
        scenarios with large expected reward rates should have stronger priors and
        tend towards larger random numbers. Also, the more arms there are, the smaller
        the optimal expected value for these random numbers.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (streaming),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    refit_buffer : int or None
        Number of observations per arm to keep as a reserve for passing to
        'partial_fit'. If passing it, up until the moment there are at least this
        number of observations for a given arm, that arm will keep the observations
        when calling 'fit' and 'partial_fit', and will translate calls to
        'partial_fit' to calls to 'fit' with the new plus stored observations.
        After the reserve number is reached, calls to 'partial_fit' will enlarge
        the data batch with the stored observations, and old stored observations
        will be gradually replaced with the new ones (at random, not on a FIFO
        basis). This technique can greatly enchance the performance when fitting
        the data in batches, but memory consumption can grow quite large.
        If passing sparse CSR matrices as input to 'fit' and 'partial_fit',
        these will be converted to dense once they go into this reserve, and
        then converted back to CSR to augment the new data.
        Calls to 'fit' will override this reserve.
        Ignored when passing 'batch_train=False'.
    deep_copy_buffer : bool
        Whether to make deep copies of the data that is stored in the
        reserve for ``refit_buffer``. If passing 'False', when the reserve is
        not yet full, these will only store shallow copies of the data, which
        is faster but will not let Python's garbage collector free memory
        after deleting the data, and if the original data is overwritten, so will
        this buffer.
        Ignored when not using ``refit_buffer``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Note that if the base algorithm is itself parallelized,
        this might result in a slowdown as both compete for available threads, so don't set
        parallelization in both. The parallelization uses shared memory, thus you will only
        see a speed up if your base classifier releases the Python GIL, and will
        otherwise result in slower runs.

    References
    ----------
    .. [1] Cortes, David. "Adapting multi-armed bandits policies to contextual bandits scenarios."
           arXiv preprint arXiv:1811.04383 (2018).
    """
    def __init__(self, base_algorithm, nchoices, multiplier=1.0, inflation_rate=1.0004,
                 beta_prior='auto', smoothing=None, noise_to_smooth=True,
                 batch_train=False, refit_buffer=None, deep_copy_buffer=True,
                 assume_unique_reward=False, random_state=None, njobs=-1):
        self._add_common_params(base_algorithm, beta_prior, smoothing, noise_to_smooth, njobs, nchoices,
                                batch_train, refit_buffer, deep_copy_buffer,
                                assume_unique_reward, random_state)

        if multiplier is not None:
            if isinstance(multiplier, int):
                multiplier = float(multiplier)
            assert multiplier > 0
        else:
            multiplier = None
        if inflation_rate is not None:
            if isinstance(inflation_rate, int):
                inflation_rate = float(inflation_rate)
            assert inflation_rate > 0
        self.multiplier = multiplier
        self.inflation_rate = inflation_rate

    def reset_multiplier(self, multiplier=1.0):
        """
        Set the multiplier to a custom number

        Parameters
        ----------
        multiplier : float
            New multiplier for the numbers going to the softmax function.
            Note that it will still apply the inflation rate after this
            parameter is being reset.

        Returns
        -------
        self : obj
            This object
        """
        assert multiplier != 0
        self.multiplier = multiplier
        return self
    
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
        X = _check_X_input(X)
        if not self.is_fitted:
            raise ValueError("Object has not been fit to data.")
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
        pred : array (n_samples,) or dict("choice" : array(n_samples,), "score" : array(n_samples,))
            Actions chosen by the policy. If passing output_score=True, it will be a dictionary
            with the chosen arm and the score that the arm got following this policy with the classifiers used.
        """
        if not self.is_fitted:
            return self._predict_random_if_unfit(X, output_score)
        if exploit:
            X = _check_X_input(X)
            return np.argmax(self._oracles.decision_function(X), axis=1)
        pred = self._softmax_scores(X)
        chosen =  _choice_over_rows(pred, self.random_state, self.njobs)

        if output_score:
            score_chosen = pred[np.arange(pred.shape[0]), chosen]
        chosen = self._name_arms(chosen)

        if not output_score:
            return chosen
        else:
            return {"choice" : chosen, "score" : score_chosen}

    def _softmax_scores(self, X):
        pred = self.decision_function(X)
        _apply_inverse_sigmoid(pred)
        if self.multiplier is not None:
            pred *= self.multiplier
            if self.inflation_rate is not None:
                self.multiplier *= self.inflation_rate ** pred.shape[0]
        _apply_softmax(pred)
        return pred

    def topN(self, X, n):
        """
        Get top-N ranked actions for each observation

        Note
        ----
        This method will rank choices/arms according to what the policy
        dictates - it is not an exploitation-mode rank, so if e.g. there are
        random choices for some observations, there will be random ranks in here.

        Parameters
        ----------
        X : array (n_samples, n_features)
            New observations for which to rank actions according to this policy.
        n : int
            Number of top-ranked actions to output

        Returns
        -------
        topN : array(n_samples, n)
            The top-ranked actions for each observation
        """
        assert n >= 1
        if isinstance(n, float):
            n = int(n)
        assert isinstance(n, int)
        if n > self.nchoices:
            raise ValueError("'n' cannot be greater than 'nchoices'.")
        X = _check_X_input(X)

        scores = self._softmax_scores(X)
        topN = topN_byrow_softmax(scores, n, self.njobs, self.random_state)
        return self._name_arms(topN)


class LinUCB(_BasePolicyWithExploit):
    """
    LinUCB

    Note
    ----
    This strategy requires each fitted model to store a square matrix with
    dimension equal to the number of features. Thus, memory consumption can grow
    very high with this method.

    Note
    ----
    The 'X' data (covariates) should ideally be centered before passing them
    to 'fit', 'partial_fit', 'predict'.

    Note
    ----
    The default hyperparameters here are meant to match the original reference, but
    it's recommended to change them. Particularly: use ``beta_prior`` instead of
    ``ucb_from_empty``, decrease ``alpha``, and maybe increase ``lambda_``.
    
    Parameters
    ----------
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    alpha : float
        Parameter to control the upper confidence bound (more is higher).
    lambda_ : float > 0
        Regularization parameter. References assumed this would always be equal to 1, but this
        implementation allows to change it.
    fit_intercept : bool
        Whether to add an intercept term to the coefficients.
    use_float : bool
        Whether to use C 'float' type for the required matrices. If passing 'False',
        will use C 'double'. Be aware that memory usage for this model can grow
        very large.
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
    ucb_from_empty : bool
        Whether to make upper confidence bounds on arms with no observations according
        to the formula, as suggested in the references (ties are broken at random for
        them). Choosing this option leads to policies that usually start making random
        predictions until having sampled from all arms, and as such, it's not
        recommended when the number of arms is large relative to the number of rounds.
        Instead, it's recommended to use ``beta_prior``, which acts in the same way
        as for the other policies in this library.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((3/log2(nchoices), 4), 2).
        This parameter can have a very large impact in the end results, and it's
        recommended to tune it accordingly - scenarios with low expected reward rates
        should have priors that result in drawing small random numbers, whereas
        scenarios with large expected reward rates should have stronger priors and
        tend towards larger random numbers. Also, the more arms there are, the smaller
        the optimal expected value for these random numbers.
        Ignored when passing ``ucb_from_empty=True``.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
        Note that it is technically incorrect to apply smoothing like this (because
        the predictions from models are not bounded between zero and one), but
        if neither ``beta_prior``, nor ``smoothing`` are passed, the policy can get
        stuck in situations in which it will only choose actions from the first batch
        of observations to which it is fit (if using ``ucb_from_empty=False``), or
        only from the first arms that show rewards (if using ``ucb_from_empty=True``).
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Be aware that the algorithm will use BLAS function calls,
        and if these have multi-threading enabled, it might result in a slow-down
        as both functions compete for available threads.
    
    References
    ----------
    .. [1] Chu, Wei, et al. "Contextual bandits with linear payoff functions."
           Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. 2011.
    .. [2] Li, Lihong, et al. "A contextual-bandit approach to personalized news article recommendation."
           Proceedings of the 19th international conference on World wide web. ACM, 2010.
    """
    def __init__(self, nchoices, alpha=1.0, lambda_=1.0, fit_intercept=True,
                 use_float=True, method="sm", ucb_from_empty=True,
                 beta_prior=None, smoothing=None, noise_to_smooth=True,
                 assume_unique_reward=False, random_state=None, njobs=1):
        self._ts = False
        self._add_common_lin(alpha, lambda_, fit_intercept, use_float, method, nchoices, njobs)
        base = _LinUCB_n_TS_single(alpha=self.alpha, lambda_=self.lambda_,
                                   fit_intercept=self.fit_intercept,
                                   use_float=self.use_float, method=self.method,
                                   ts=False)
        self._add_common_params(base, beta_prior, smoothing, noise_to_smooth, njobs, nchoices,
                                True, None, False, assume_unique_reward,
                                random_state, assign_algo=True, prior_def_ucb=True,
                                force_unfit_predict=ucb_from_empty)

    def _add_common_lin(self, alpha, lambda_, fit_intercept, use_float, method, nchoices, njobs):
        if isinstance(alpha, int):
            alpha = float(alpha)
        assert isinstance(alpha, float)
        if isinstance(lambda_, int):
            lambda_ = float(lambda_)
        assert lambda_ >= 0.
        assert method in ["chol", "sm"]

        self.alpha = alpha
        self.lambda_ = lambda_
        self.fit_intercept = bool(fit_intercept)
        self.use_float = bool(use_float)
        self.method = method
        if self._ts:
            self.v_sq = self.alpha
            del self.alpha

    def reset_alpha(self, alpha=1.0):
        """
        Set the upper confidence bound parameter to a custom number

        Note
        ----
        This method is only for LinUCB, not for LinTS.

        Parameters
        ----------
        alpha : float
            Parameter to control the upper confidence bound (more is higher).

        Returns
        -------
        self : obj
            This object
        """
        if self._ts:
            raise ValueError("Method is only available for LinUCB")
        if isinstance(alpha, int):
            alpha = float(alpha)
        assert isinstance(alpha, float)
        self.alpha = alpha
        self.base_algorithm.alpha = alpha
        if self.is_fitted:
            self._oracles.reset_attribute("alpha", alpha)
        return self

class LinTS(LinUCB):
    """
    Linear Thompson Sampling

    Note
    ----
    This strategy requires each fitted model to store a square matrix with
    dimension equal to the number of features. Thus, memory consumption can grow
    very high with this method.

    Note
    ----
    The 'X' data (covariates) should ideally be centered before passing them
    to 'fit', 'partial_fit', 'predict'.

    Note
    ----
    Be aware that sampling coefficients is an operation that scales poorly with
    the number of columns/features/variables. For wide datasets, it might be
    slower than a bootstrapped approach, especially when using ``sample_unique=True``.
    
    Parameters
    ----------
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    v_sq : float
        Parameter by which to multiply the covariance matrix (more means higher variance).
    lambda_ : float > 0
        Regularization parameter. References assumed this would always be equal to 1, but this
        implementation allows to change it.
    fit_intercept : bool
        Whether to add an intercept term to the coefficients.
    sample_unique : bool
        Whether to sample different coefficients each time a prediction is to
        be made. If passing 'False', when calling 'predict', it will sample
        the same coefficients for all the observations in the same call to
        'predict', whereas if passing 'True', will use a different set of
        coefficients for each observations. Passing 'False' leads to an
        approach which is theoretically wrong, but as sampling coefficients
        can be very slow, using 'False' can provide a reasonable speed up
        without much of a performance penalty.
    use_float : bool
        Whether to use C 'float' type for the required matrices. If passing 'False',
        will use C 'double'. Be aware that memory usage for this model can grow
        very large.
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
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((2/log2(nchoices), 4), 2)
        This parameter can have a very large impact in the end results, and it's
        recommended to tune it accordingly - scenarios with low expected reward rates
        should have priors that result in drawing small random numbers, whereas
        scenarios with large expected reward rates should have stronger priors and
        tend towards larger random numbers. Also, the more arms there are, the smaller
        the optimal expected value for these random numbers.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
        Note that it is technically incorrect to apply smoothing like this (because
        the predictions from models are not bounded between zero and one), but
        if neither ``beta_prior``, nor ``smoothing`` are passed, the policy can get
        stuck in situations in which it will only choose actions from the first batch
        of observations to which it is fit.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Be aware that the algorithm will use BLAS function calls,
        and if these have multi-threading enabled, it might result in a slow-down
        as both functions compete for available threads.
    
    References
    ----------
    .. [1] Agrawal, Shipra, and Navin Goyal.
           "Thompson sampling for contextual bandits with linear payoffs."
           International Conference on Machine Learning. 2013.
    """
    def __init__(self, nchoices, v_sq=1.0, lambda_=1.0, fit_intercept=True,
                 sample_unique=False, use_float=True, method="sm",
                 beta_prior=None, smoothing=None, noise_to_smooth=True,
                 assume_unique_reward=False, random_state=None, njobs = 1):
        self._ts = True
        self._add_common_lin(v_sq, lambda_, fit_intercept, use_float, method, nchoices, njobs)
        base = _LinUCB_n_TS_single(alpha=self.v_sq, lambda_=self.lambda_,
                                   fit_intercept=self.fit_intercept,
                                   use_float=self.use_float, method=self.method,
                                   ts=True, sample_unique=sample_unique)
        self._add_common_params(base, beta_prior, smoothing, noise_to_smooth, njobs, nchoices,
                                True, None, False, assume_unique_reward,
                                random_state, assign_algo=True, prior_def_ucb=False)

    def reset_v_sq(self, v_sq=1.0):
        """
        Set the covariance multiplier to a custom number

        Parameters
        ----------
        v_sq : float
            Parameter by which to multiply the covariance matrix (more means higher variance).

        Returns
        -------
        self : obj
            This object
        """
        if isinstance(v_sq, int):
            v_sq = float(v_sq)
        assert isinstance(v_sq, float)
        self.v_sq = v_sq
        self.base_algorithm.alpha = v_sq
        if self.is_fitted:
            self._oracles.reset_attribute("alpha", v_sq)
        return self

class ParametricTS(_BasePolicyWithExploit):
    """
    Parametric Thompson Sampling

    Performs Thompson sampling using a beta distribution, with parameters given
    by the predicted probability from the base algorithm multiplied by the number
    of observations seen from each arm.

    Parameters
    ----------
    base_algorithm : obj
        Base binary classifier for which each sample for each class will be fit.
        Will look for, in this order:
            1) A 'predict_proba' method with outputs (n_samples, 2), values in [0,1], rows suming to 1
            2) A 'decision_function' method with unbounded outputs (n_samples,) to which it will apply a sigmoid function.
            3) A 'predict' method with outputs (n_samples,) with values in [0,1].
        Can also pass a list with a different (or already-fit) classifier for each arm.
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((2/log2(nchoices), 4), 2)
        This parameter can have a very large impact in the end results, and it's
        recommended to tune it accordingly - scenarios with low expected reward rates
        should have priors that result in drawing small random numbers, whereas
        scenarios with large expected reward rates should have stronger priors and
        tend towards larger random numbers. Also, the more arms there are, the smaller
        the optimal expected value for these random numbers.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    beta_prior_ts : tuple(float, float)
        Beta prior used for the distribution from which to draw probabilities given
        the base algorithm's estimates. This is independent of ``beta_prior``, and
        they will not be used together under the same arm. Pass '(0,0)' for no prior.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        This will not work well with non-probabilistic classifiers such as SVM, in which case you might
        want to define a class that embeds it with some recalibration built-in.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    batch_train : bool
        Whether the base algorithm will be fit to the data in batches as it comes (streaming),
        or to the whole dataset each time it is refit. Requires a classifier with a
        'partial_fit' method.
    refit_buffer : int or None
        Number of observations per arm to keep as a reserve for passing to
        'partial_fit'. If passing it, up until the moment there are at least this
        number of observations for a given arm, that arm will keep the observations
        when calling 'fit' and 'partial_fit', and will translate calls to
        'partial_fit' to calls to 'fit' with the new plus stored observations.
        After the reserve number is reached, calls to 'partial_fit' will enlarge
        the data batch with the stored observations, and old stored observations
        will be gradually replaced with the new ones (at random, not on a FIFO
        basis). This technique can greatly enchance the performance when fitting
        the data in batches, but memory consumption can grow quite large.
        If passing sparse CSR matrices as input to 'fit' and 'partial_fit',
        these will be converted to dense once they go into this reserve, and
        then converted back to CSR to augment the new data.
        Calls to 'fit' will override this reserve.
        Ignored when passing 'batch_train=False'.
    deep_copy_buffer : bool
        Whether to make deep copies of the data that is stored in the
        reserve for ``refit_buffer``. If passing 'False', when the reserve is
        not yet full, these will only store shallow copies of the data, which
        is faster but will not let Python's garbage collector free memory
        after deleting the data, and if the original data is overwritten, so will
        this buffer.
        Ignored when not using ``refit_buffer``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
        While this controls random number generation for this meteheuristic,
        there can still be other sources of variations upon re-runs, such as
        data aggregations in parallel (e.g. from OpenMP or BLAS functions).
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Note that if the base algorithm is itself parallelized,
        this might result in a slowdown as both compete for available threads, so don't set
        parallelization in both. The parallelization uses shared memory, thus you will only
        see a speed up if your base classifier releases the Python GIL, and will
        otherwise result in slower runs.
    """
    def __init__(self, base_algorithm, nchoices, beta_prior=None,
                 beta_prior_ts=(0.,0.), smoothing=None, noise_to_smooth=True,
                 batch_train=False, refit_buffer=None, deep_copy_buffer=True,
                 assume_unique_reward=False, random_state=None, njobs=-1):
        self._add_common_params(base_algorithm, beta_prior, smoothing, noise_to_smooth, njobs, nchoices,
                                batch_train, refit_buffer, deep_copy_buffer,
                                assume_unique_reward, random_state)
        assert beta_prior_ts[0] >= 0.
        assert beta_prior_ts[1] >= 0.
        self.beta_prior_ts = beta_prior_ts
        self.force_counters = True

    def reset_beta_prior_ts(self, beta_prior_ts=(0.,0.)):
        """
        Set the Thompson prior to a custom tuple

        Parameters
        ----------
        beta_prior_ts : tuple(float, float)
            Beta prior used for the distribution from which to draw probabilities given
            the base algorithm's estimates. This is independent of ``beta_prior``, and
            they will not be used together under the same arm. Pass '(0,0)' for no prior.

        Returns
        -------
        self : obj
            This object
        """
        assert beta_prior_ts[0] >= 0.
        assert beta_prior_ts[1] >= 0.
        self.beta_prior_ts = beta_prior_ts
        return self

    def _score_matrix(self, X):
        pred = self._oracles.decision_function(X)
        counters = self._oracles.get_nobs_by_arm()
        with_model = counters >= self.beta_prior[1]
        counters = counters.reshape((1,-1))
        pred[:, with_model] = self.random_state.beta(
            np.clip(pred[:, with_model] * counters[:, with_model] + self.beta_prior_ts[0], a_min=1e-5, a_max=None),
            np.clip((1. - pred[:, with_model]) * counters[:, with_model] + self.beta_prior_ts[1], a_min=1e-5, a_max=None)
            )
        return pred

    def _exploit(self, X):
        return self._oracles.decision_function(X)

class PartitionedUCB(_BasePolicyWithExploit):
    """
    Tree-partitioned Upper Confidence Bound

    Fits decision trees having non-contextual multi-armed UCB bandits at each leaf.
    Uses the standard approximation for confidence interval of a proportion
    (mean + c * sqrt(mean * (1-mean) / n)).

    This is similar to the 'TreeHeuristic' in the reference paper, but uses UCB as a
    MAB policy instead of Thompson sampling.

    Note
    ----
    This method fits only one tree per arm. As such, it's not recommended for
    high-dimensional data.

    Parameters
    ----------
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    percentile : int [0,100]
        Percentile of the confidence interval to take.
    ucb_prior : tuple(float, float)
        Prior for the upper confidence bounds generated at each tree leaf. First
        number will be added to the number of positives, and second number to
        the number of negatives. If passing ``beta_prior=None``, will use these alone
        to generate an upper confidence bound and will break ties at random.
    beta_prior : str 'auto', None, or tuple ((a,b), n)
        If not 'None', when there are less than 'n' samples with and without
        a reward from a given arm, it will predict the score for that class as a
        random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'. If set to "auto", will be calculated as:
            beta_prior = ((3/log2(nchoices), 4), 2)
        This parameter can have a very large impact in the end results, and it's
        recommended to tune it accordingly - scenarios with low expected reward rates
        should have priors that result in drawing small random numbers, whereas
        scenarios with large expected reward rates should have stronger priors and
        tend towards larger random numbers. Also, the more arms there are, the smaller
        the optimal expected value for these random numbers.
        Note that this method calculates upper bounds rather than expectations, so the 'a'
        parameter should be higher than for other methods.
        Recommended to use only one of ``beta_prior`` or ``smoothing``.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        Not recommended for this method.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Note that it will not achieve a large
        degree of parallelization due to needing many Python computations with
        shared memory and no GIL releasing.
    *args : tuple
        Additional arguments to pass to the decision tree model (this policy uses
        SciKit-Learn's ``DecisionTreeClassifier`` - see their docs for more details).
        Note that passing ``random_state`` for ``DecisionTreeClassifier`` will have
        no effect as it will be set independently.
    **kwargs : dict
        Additional keyword arguments to pass to the decision tree model (this policy uses
        SciKit-Learn's ``DecisionTreeClassifier`` - see their docs for more details).
        Note that passing ``random_state`` for ``DecisionTreeClassifier`` will have
        no effect as it will be set independently.
    
    References
    ----------
    .. [1] Elmachtoub, Adam N., et al.
           "A practical method for solving contextual bandit problems using decision trees."
           arXiv preprint arXiv:1706.04687 (2017).
    .. [2] https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """
    def __init__(self, nchoices, percentile=80, ucb_prior=(1,1),
                 beta_prior='auto', smoothing=None, noise_to_smooth=True,
                 assume_unique_reward=False, random_state=None, njobs=-1,
                 *args, **kwargs):
        assert (percentile > 0) and (percentile < 100)
        assert ucb_prior[0] >= 0.
        assert ucb_prior[1] >= 0.
        self.ucb_prior = (float(ucb_prior[0]), float(ucb_prior[1]))

        base = _TreeUCB_n_TS_single(self.ucb_prior, ts=False, alpha=float(percentile),
                                    random_state=None, *args, **kwargs)
        self._add_common_params(base, beta_prior, smoothing, noise_to_smooth, njobs,
                                nchoices, False, None, False,
                                assume_unique_reward, random_state,
                                prior_def_ucb = True,
                                force_unfit_predict = beta_prior is None)
        if self.beta_prior[1] <= 0:
            self.force_unfit_predict = True

    def reset_percentile(self, percentile=80):
        """
        Set the upper confidence bound percentile to a custom number

        Parameters
        ----------
        percentile : int [0,100]
            Percentile of the confidence interval to take.

        Returns
        -------
        self : obj
            This object
        """
        assert (percentile > 0) and (percentile < 100)
        if self.is_fitted:
            self._oracles.reset_attribute("alpha", percentile)
        self.base_algorithm.alpha = percentile
        return self

    def reset_ucb_prior(self, ucb_prior=(1,1)):
        """
        Set the upper confidence bound prior to a custom tuple

        Parameters
        ----------
        ucb_prior : tuple(float, float)
            Prior for the upper confidence bounds generated at each tree leaf. First
            number will be added to the number of positives, and second number to
            the number of negatives. If passing ``beta_prior=None``, will use these alone
            to generate an upper confidence bound and will break ties at random.

        Returns
        -------
        self : obj
            This object
        """
        assert ucb_prior[0] >= 0.
        assert ucb_prior[1] >= 0.
        self.ucb_prior = (float(ucb_prior[0]), float(ucb_prior[1]))
        self.base_algorithm.beta_prior = ucb_prior
        if self.is_fitted:
            self._oracles.reset_attribute("beta_prior", ucb_prior)
        return self

class PartitionedTS(_BasePolicyWithExploit):
    """
    Tree-partitioned Thompson Sampling

    Fits decision trees having non-contextual multi-armed Thompson-sampling
    bandits at each leaf.

    This corresponds to the 'TreeHeuristic' in the reference paper.

    Note
    ----
    This method fits only one tree per arm. As such, it's not recommended for
    high-dimensional data.

    Note
    ----
    The default values for beta prior are as suggested in the reference paper.
    It is recommended to change it however.

    Parameters
    ----------
    nchoices : int or list-like
        Number of arms/labels to choose from. Can also pass a list, array, or Series with arm names, in which case
        the outputs from predict will follow these names and arms can be dropped by name, and new ones added with a
        custom name.
    beta_prior : str 'auto', or tuple ((a,b), n)
        When there are less than 'n' samples with and without a reward from
        a given arm, it will predict the score
        for that class as a random number drawn from a beta distribution with the prior
        specified by 'a' and 'b'.
        If passing 'auto' (which is *not* the default), will use the same default as for
        the other policies in this library:
            beta_prior = ((2/log2(nchoices), 4), 2)
        Additionally, will use (a,b) as prior when sampling from the MAB at a given node.
    smoothing : None, tuple (a,b), or list
        If not None, predictions will be smoothed as yhat_smooth = (yhat*n + a)/(n + b),
        where 'n' is the number of times each arm was chosen in the training data.
        Can also pass it as a list of tuples with different 'a' and 'b' parameters for each arm
        (e.g. if there are arm features, these parameters can be determined through a different model).
        Not recommended for this method.
    noise_to_smooth : bool
        If passing ``smoothing``, whether to add a small amount of random
        noise ~ Uniform(0, 10^-12) in order to break ties at random instead of
        choosing the smallest arm index.
        Ignored when passing ``smoothing=None``.
    assume_unique_reward : bool
        Whether to assume that only one arm has a reward per observation. If set to 'True',
        whenever an arm receives a reward, the classifiers for all other arms will be
        fit to that observation too, having negative label.
    random_state : int, None, RandomState, or Generator
        Either an integer which will be used as seed for initializing a
        ``Generator`` object for random number generation, a ``RandomState``
        object (from NumPy) from which to draw an integer, or a ``Generator``
        object (from NumPy), which will be used directly.
    njobs : int or None
        Number of parallel jobs to run. If passing None will set it to 1. If passing -1 will
        set it to the number of CPU cores. Note that it will not achieve a large
        degree of parallelization due to needing many Python computations with
        shared memory and no GIL releasing.
    *args : tuple
        Additional arguments to pass to the decision tree model (this policy uses
        SciKit-Learn's ``DecisionTreeClassifier`` - see their docs for more details).
        Note that passing ``random_state`` for ``DecisionTreeClassifier`` will have
        no effect as it will be set independently.
    **kwargs : dict
        Additional keyword arguments to pass to the decision tree model (this policy uses
        SciKit-Learn's ``DecisionTreeClassifier`` - see their docs for more details).
        Note that passing ``random_state`` for ``DecisionTreeClassifier`` will have
        no effect as it will be set independently.

    References
    ----------
    .. [1] Elmachtoub, Adam N., et al.
           "A practical method for solving contextual bandit problems using decision trees."
           arXiv preprint arXiv:1706.04687 (2017).
    .. [2] https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """
    def __init__(self, nchoices, beta_prior=((1,1), 1), smoothing=None, noise_to_smooth=True,
                 assume_unique_reward=False, random_state=None, njobs=-1,
                 *args, **kwargs):
        if beta_prior is None:
            raise ValueError("Must pass a valid 'beta_prior'.")
        beta_prior = _check_beta_prior(beta_prior, nchoices)
        base = _TreeUCB_n_TS_single(beta_prior[0], ts=True, random_state=None,
                                    *args, **kwargs)
        self._add_common_params(base, beta_prior, smoothing, noise_to_smooth, njobs,
                                nchoices, False, None, False,
                                assume_unique_reward, random_state)
