# Tests for issue #69: support for non-binary / continuous rewards in [0, 1] with the
# regressor-backed policies (LinUCB and bootstrapped policies whose base estimator is a
# plain regressor).
#
# The "works with continuous [0, 1] rewards" set is the policies whose exploration does
# not assume binary rewards: the built-in linear models score arms from the regression's
# predictions, and the bootstrapped policies explore via resampling. These are used with
# ``beta_prior=None`` and ``smoothing=None`` (the only options that introduce a [0, 1]
# dependence). See the "Non-binary / continuous rewards" section of the documentation.
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from contextualbandits.online import BootstrappedUCB, LinUCB

NCHOICES = 4
NFEATURES = 5
SEED = 111


def _make_world(seed):
    """Return a function mapping X -> (n, nchoices) matrix of expected rewards in (0, 1)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(NCHOICES, NFEATURES)) * 1.5
    b = rng.normal(size=NCHOICES) * 1.5
    return lambda X: 1.0 / (1.0 + np.exp(-(X @ W.T + b)))


def _sample_continuous_reward(rng, expected_reward, X, a):
    """Observed reward = expected reward + small clipped noise -> continuous values in [0, 1]."""
    mu = expected_reward(X)[np.arange(X.shape[0]), a]
    return np.clip(mu + rng.normal(scale=0.03, size=mu.shape[0]), 0.0, 1.0)


def _fit_on_random_logging_policy(make_policy, seed, n_train=4000):
    """Fit a policy on data collected by a uniform-random logging policy.

    Using a random logging policy guarantees every arm is well covered and there is no
    exploration feedback loop, so the test exercises whether the base model *learns* from
    continuous rewards rather than the exploration heuristic's tendency to get stuck (which
    is an orthogonal, already-documented behavior of ``beta_prior=None``).
    """
    expected_reward = _make_world(seed)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train, NFEATURES))
    a = rng.integers(0, NCHOICES, size=n_train)
    r = _sample_continuous_reward(rng, expected_reward, X, a)

    # Sanity: rewards are genuinely continuous and within [0, 1], not {0, 1}.
    assert ((r >= 0.0) & (r <= 1.0)).all()
    assert np.unique(r).shape[0] > 2

    pol = make_policy()
    pol.fit(X, a, r)
    return pol, expected_reward, rng


def _assert_policy_beats_random(pol, expected_reward, rng):
    """Assert the fitted policy earns more reward than a uniform-random baseline.

    Beating the random baseline is the version-independent property: it holds whenever
    the policy has learned anything from the continuous rewards. Oracle agreement is kept
    only as a loose secondary check against the 1 / NCHOICES chance rate, not pinned to an
    observed value (which would not be reproducible across platforms / library versions).
    """
    X_eval = rng.normal(size=(4000, NFEATURES))
    oracle = expected_reward(X_eval).argmax(axis=1)
    chosen = np.asarray(pol.predict(X_eval)).astype(int)

    r_pol = _sample_continuous_reward(rng, expected_reward, X_eval, chosen)
    r_rand = _sample_continuous_reward(
        rng, expected_reward, X_eval, rng.integers(0, NCHOICES, size=X_eval.shape[0])
    )

    assert r_pol.mean() > r_rand.mean()
    assert float((chosen == oracle).mean()) > 1.0 / NCHOICES  # better than chance


def test_linucb_continuous_rewards():
    pol, expected_reward, rng = _fit_on_random_logging_policy(
        lambda: LinUCB(nchoices=NCHOICES, beta_prior=None, smoothing=None, random_state=SEED),
        SEED,
    )
    _assert_policy_beats_random(pol, expected_reward, rng)


@pytest.mark.parametrize("base", [LinearRegression, Ridge])
def test_bootstrapped_ucb_regressor_base_continuous_rewards(base):
    pol, expected_reward, rng = _fit_on_random_logging_policy(
        lambda: BootstrappedUCB(
            base(), nchoices=NCHOICES, beta_prior=None, smoothing=None, random_state=SEED
        ),
        SEED,
    )
    _assert_policy_beats_random(pol, expected_reward, rng)


def test_linucb_partial_fit_continuous_rewards():
    """Streaming updates with continuous rewards run without error and yield valid choices."""
    rng = np.random.default_rng(SEED)
    pol = LinUCB(nchoices=NCHOICES, beta_prior=None, smoothing=None, random_state=SEED)
    for _ in range(10):
        X_batch = rng.normal(size=(100, NFEATURES))
        a_batch = rng.integers(0, NCHOICES, size=100)
        r_batch = rng.random(100)  # continuous rewards already in [0, 1)
        pol.partial_fit(X_batch, a_batch, r_batch)
    pred = np.asarray(pol.predict(rng.normal(size=(20, NFEATURES)))).astype(int)
    assert pred.shape == (20,)
    assert ((pred >= 0) & (pred < NCHOICES)).all()
