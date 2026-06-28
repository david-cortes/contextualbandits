# Tests about using a plain regressor (an estimator with only a ``.predict``
# method, e.g. ``sklearn.linear_model.LinearRegression`` / ``Ridge``) as the base
# estimator, with binary rewards r in {0, 1}.
#
# ActiveExplorer is included as a negative case: its active-learning step needs gradients,
# so with the default ``f_grad_norm="auto"`` it rejects an arbitrary regressor.
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from contextualbandits.online import (
    AdaptiveGreedy,
    ActiveExplorer,
    BootstrappedUCB,
    EpsilonGreedy,
    SeparateClassifiers,
)

NCHOICES = 4
NFEATURES = 5
SEED = 111


def _make_world(seed):
    """Return a function mapping X -> (n, nchoices) matrix of reward probabilities in (0, 1)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(NCHOICES, NFEATURES)) * 1.5
    b = rng.normal(size=NCHOICES) * 1.5
    return lambda X: 1.0 / (1.0 + np.exp(-(X @ W.T + b)))


def _sample_binary_reward(rng, reward_prob, X, a):
    """Observed reward is a Bernoulli draw from P(r=1|x,arm) -> genuinely binary {0, 1}."""
    p = reward_prob(X)[np.arange(X.shape[0]), a]
    return (rng.random(p.shape[0]) < p).astype(int)


def _fit_on_random_logging_policy(make_policy, seed, n_train=4000):
    """Fit a policy on data collected by a uniform-random logging policy.

    A random logging policy guarantees every arm is well covered and there is no
    exploration feedback loop, so the test isolates whether the regressor base *learns*
    from the binary rewards rather than any exploration heuristic's behavior.
    """
    reward_prob = _make_world(seed)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train, NFEATURES))
    a = rng.integers(NCHOICES, size=n_train)
    r = _sample_binary_reward(rng, reward_prob, X, a)

    # Sanity: rewards are genuinely binary {0, 1}, both classes present.
    assert set(np.unique(r)).issubset({0.0, 1.0})
    assert np.unique(r).shape[0] == 2

    pol = make_policy()
    pol.fit(X, a, r)
    return pol, reward_prob, rng


def _greedy_choice(pol, X):
    return np.asarray(pol.predict(X))


def _assert_policy_beats_random(pol, reward_prob, rng):
    """Assert the fitted policy earns more *expected* reward than a uniform-random baseline.

    Comparing expected reward (the true P(r=1|x,arm)) of the chosen vs random arms is the
    version-independent, low-variance property: it holds whenever the regressor base has
    learned the arm structure. Oracle agreement is kept only as a loose check against the
    1 / NCHOICES chance rate, not pinned to an observed value.
    """
    X_eval = rng.normal(size=(4000, NFEATURES))
    p = reward_prob(X_eval)
    oracle = p.argmax(axis=1)
    chosen = _greedy_choice(pol, X_eval)

    exp_reward_chosen = p[np.arange(X_eval.shape[0]), chosen].mean()
    exp_reward_random = p.mean()  # expected reward of a uniform-random arm

    assert exp_reward_chosen > exp_reward_random
    assert float((chosen == oracle).mean()) > 1.0 / NCHOICES  # better than chance


class _OutOfRangeRegressor:
    """A scikit-learn-style regressor whose predictions land outside [0, 1] with high
    probability, used to exercise the unbounded-output path the docstrings describe.

    It fits an ordinary ``LinearRegression`` on the (binary) rewards and shifts every
    prediction by a fixed positive offset. The shift is identical across arms, so it
    preserves each policy's arm ranking (argmax over predicted reward) — and thus the
    learned structure — while guaranteeing the base estimator emits values well outside
    [0, 1]. The offset is kept positive so the predictions stay above zero, matching the
    documented caveat for threshold-based policies (e.g. ``AdaptiveGreedy``).
    """

    def __init__(self, offset=10.0):
        self.offset = offset

    def fit(self, X, y, sample_weight=None):
        self._base = LinearRegression()
        self._base.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._base.predict(X) + self.offset

    def get_params(self, deep=True):
        return {"offset": self.offset}

    def set_params(self, **params):
        if "offset" in params:
            self.offset = params["offset"]
        return self


def _make(cls, base):
    return lambda: cls(base(), nchoices=NCHOICES, beta_prior=None, smoothing=None, random_state=SEED)


@pytest.mark.parametrize(
    "cls",
    [BootstrappedUCB, SeparateClassifiers, EpsilonGreedy, AdaptiveGreedy],
)
def test_regressor_base_binary_rewards(cls):
    pol, reward_prob, rng = _fit_on_random_logging_policy(_make(cls, LinearRegression), SEED)
    _assert_policy_beats_random(pol, reward_prob, rng)


def test_regressor_base_ridge():
    """A different regressor base (Ridge) also learns under binary rewards."""
    pol, reward_prob, rng = _fit_on_random_logging_policy(_make(BootstrappedUCB, Ridge), SEED)
    _assert_policy_beats_random(pol, reward_prob, rng)


def test_regressor_base_predictions_outside_unit_interval():
    """The base regressor may output values outside [0, 1]; policies that rank arms by
    predicted reward still learn, because an order-preserving shift leaves the per-arm
    argmax unchanged. This guards the documented "ideally in [0,1] but should also work
    with regressors mostly around [0,1]" behavior by actually exercising the out-of-range
    path, rather than relying on a [0,1]-fit LinearRegression that rarely leaves the unit
    interval.
    """
    # Sanity: the made-up base genuinely predicts outside [0, 1] with high probability,
    # so the policies below are exercised on unbounded inputs (not just incidentally).
    reward_prob = _make_world(SEED)
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(2000, NFEATURES))
    a = rng.integers(0, NCHOICES, size=2000)
    r = _sample_binary_reward(rng, reward_prob, X, a)
    preds = _OutOfRangeRegressor().fit(X, r).predict(X)
    assert np.mean((preds < 0.0) | (preds > 1.0)) > 0.9

    # Argmax-ranking policies still beat random under the order-preserving shift.
    for cls in (BootstrappedUCB, SeparateClassifiers, EpsilonGreedy, AdaptiveGreedy):
        pol, reward_prob_i, rng_i = _fit_on_random_logging_policy(
            _make(cls, _OutOfRangeRegressor), SEED
        )
        _assert_policy_beats_random(pol, reward_prob_i, rng_i)


def test_active_explorer_rejects_plain_regressor():
    """ActiveExplorer needs gradients; the default f_grad_norm='auto' rejects an arbitrary
    regressor such as Ridge (its class name is not in the supported-gradient whitelist)."""
    with pytest.raises(ValueError):
        ActiveExplorer(Ridge(), nchoices=NCHOICES, beta_prior=None, smoothing=None)
