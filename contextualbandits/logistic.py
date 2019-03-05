# -*- coding: utf-8 -*-

import numpy as np, warnings
from sklearn.linear_model.logistic import _logistic_loss_and_grad
from sklearn.linear_model import RidgeClassifier
from contextualbandits.utils import _check_X_input, _check_1d_inp

#### Pre-defined step size sequences
def _step_size_sqrt_over10(initial_step_size, iteration_num):
	return initial_step_size / np.sqrt(1 + int(iteration_num/10))

def _step_size_const(initial_step_size, iteration_num):
	return initial_step_size

class StochasticLogisticRegression:
	def __init__(self, reg_param=1e-3, step_size=1e-1, rmsprop_weight=0.9, rmsprop_reg=1e-4, decr_step_size=None, fit_intercept=True, random_state=1):
		"""
		Logistic Regression fit in batches to binary labels with either AdaGrad or RMSProp formulae

		Parameters
		----------
		reg_param : float
			Strength of l2 regularization. Note that the loss function has an average log-loss over observations,
			so the optimal regulatization will likely be a lot smaller than for scikit-learn's (which uses sum instead).
		step_size : float
			Initial step size to use. Note that the step might be decreased as iterations increase through function
			passed in 'decr_step_size'.
		rmsprop_weight : float(0, 1) or None
			Weight for old gradients in RMSProp formula. If passing None, will use AdaGrad formula instead.
		rmsprop_reg : float > 0
			Regularization for the square root of squared gradient sums.
		decr_step_size : None, str 'auto', or function(initial_step_size, iteration_number) -> float
			Function that determines the step size to take at each iteration.
			If passing None, will use constant step size.
			If passing 'auto', will use 1 / sqrt(1 + int(iteration_num/10) ).
			Note that the iteration numbers start at zero.
		fit_intercept : bool
			Whether to add an intercept to the model parameters.
		random_state : int
			Random seed to use.
		"""
		assert reg_param >= 0
		assert step_size > 0
		if rmsprop_weight is not None:
			assert 0 < rmsprop_weight < 1
		assert rmsprop_reg > 0
		if decr_step_size is not None:
			if decr_step_size == "auto":
				decr_step_size = _step_size_sqrt_over10
			else:
				assert callable(decr_step_size)
		else:
			decr_step_size = _step_size_const

		self.reg_param = reg_param
		self.step_size = step_size
		self.rmsprop_weight = rmsprop_weight
		self.rmsprop_reg = rmsprop_reg
		self.decr_step_size = decr_step_size
		self.fit_intercept = bool(fit_intercept)
		self.random_state = random_state

		self.is_fitted = False
		self.niter = 0
		self.n = None
		self.w = None

	def _check_inputs(self, X, y, sample_weight):
		X = _check_X_input(X)
		y = _check_1d_inp(y)
		assert X.shape[0] == y.shape[0]
		if sample_weight is None:
			sample_weight = np.ones(X.shape[0])
		assert sample_weight.shape[0] == X.shape[0]
		sample_weight /= X.shape[0]
		return X, y, sample_weight

	def _init_weights(self, n):
		np.random.seed(self.random_state)
		self.w = np.random.normal(size = n + self.fit_intercept)
		self.grad_sq_sum = np.zeros(self.w.shape[0], dtype='float64')

	def fit(self, X, y, sample_weight=None):
		"""
		Fit Logistic Regression model (will not use stochastic methods)

		Note
		----
		Calling this function will only increase the iteration numbers by 1. 

		Note
		----
		This is just a wrapper around scikit-learn's RidgeClassifier. For fitting the data
		in batches use 'partial_fit' instead.

		Parameters
		----------
		X : array(n_samples, n_features)
			Covariates (features).
		y : array(n_samples, )
			Labels for each observation (must be zero-one only).
		sample_weight : array(n_samples, ) or None
			Observation weights for each data point.

		Returns
		-------
		self : obj
			This object
		"""
		X, y, sample_weight = self._check_inputs(X, y, sample_weight)
		m = RidgeClassifier(alpha = self.reg_param, fit_intercept = self.fit_intercept)
		m.fit(X, y, sample_weight)
		if self.fit_intercept:
			self.w = np.r_[m.coef_.reshape(-1), np.array(m.intercept_).reshape(-1)]
		else:
			self.w = m.coef_.reshape(-1)
		self.is_fitted = True

		grad = _logistic_loss_and_grad(self.w, X, y, self.reg_param)[1]
		grad_sq = grad ** 2
		if self.rmsprop_weight is None:
			self.grad_sq_sum += grad_sq
		else:
			self.grad_sq_sum = self.rmsprop_weight * self.grad_sq_sum + (1 - self.rmsprop_weight) * grad_sq
		self.niter += 1
		return self

	def partial_fit(self, X, y, sample_weight=None, classes=None):
		"""
		Fit Logistic Regression model in stochastic batches

		Parameters
		----------
		X : array(n_samples, n_features)
			Covariates (features).
		y : array(n_samples, )
			Labels for each observation (must be zero-one only).
		sample_weight : array(n_samples, ) or None
			Observation weights for each data point.

		Returns
		-------
		self : obj
			This object
		"""
		X, y, sample_weight = self._check_inputs(X, y, sample_weight)
		if self.w is None:
			self._init_weights(X.shape[1])
		step_size = self.decr_step_size(self.step_size, self.niter)
		grad = _logistic_loss_and_grad(self.w, X, y, self.reg_param)[1]
		grad_sq = grad ** 2
		if self.rmsprop_weight is None:
			self.grad_sq_sum += grad_sq
		else:
			self.grad_sq_sum = self.rmsprop_weight * self.grad_sq_sum + (1 - self.rmsprop_weight) * grad_sq
		self.w -= step_size * grad / np.sqrt(self.grad_sq_sum + self.rmsprop_reg)
		
		self.niter += 1
		self.is_fitted = True
		return self

	def decision_function(self, X):
		"""
		Decision function before sigmoid transformation for new observations

		Parameters
		----------
		X : array(n_samples, n_features)
			Input data on which to predict.

		Returns
		-------
		pred : array(n_samples, )
			Raw prediction for each observation
		"""
		X = _check_X_input(X)
		if self.fit_intercept:
			return X.dot(self.w[:self.w.shape[0] - 1]) + self.w[-1]
		else:
			return X.dot(self.w)

	def predict(self, X):
		"""
		Predict the class of new observations

		Parameters
		----------
		X : array(n_samples, n_features)
			Input data on which to predict classes.

		Returns
		-------
		pred : array(n_samples, )
			Predicted class for each observation
		"""
		return (self.decision_function(X) >= 0).astype('uint8')

	def predict_proba(self, X):
		"""
		Predict class probabilities for new observations

		Parameters
		----------
		X : array(n_samples, n_features)
			Input data on which to predict class probabilities.

		Returns
		-------
		pred : array(n_samples, 2)
			Predicted class probabilities for each observation
		"""
		pred = self.decision_function(X).reshape(-1)
		pred[:] = 1 / (1 + np.exp(-pred))
		return np.c_[1 - pred, pred]

	@property
	def coef_(self):
		if not self.is_fitted:
			return None
		if self.fit_intercept:
			return self.w[:self.w.shape[0] - 1]
		else:
			return self.w

	@property
	def intercept_(self):
		if not self.is_fitted:
			return None
		if self.fit_intercept:
			return self.w[-1]
		else:
			return 0.0
