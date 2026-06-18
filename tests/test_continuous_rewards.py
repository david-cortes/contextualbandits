# Tests for issue #69: support for non-binary / continuous rewards in [0, 1] with the
# regressor-backed policies (LinUCB and bootstrapped policies whose base estimator is a
# plain regressor), and a boundary test documenting that the tree-based PartitionedUCB
# rejects continuous rewards.
#
# The "works with continuous [0, 1] rewards" set is exactly the policies whose
# exploration does not assume binary rewards: LinUCB scores arms from the regression's
# predictions plus a UCB bonus, and the bootstrapped policies explore via resampling.
# These are used with ``beta_prior=None`` and ``smoothing=None`` (the only options that
# introduce a [0, 1] dependence). See the "Non-binary / continuous rewards" section of
# the documentation.
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from contextualbandits.online import LinUCB, BootstrappedUCB, PartitionedUCB

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


def _assert_learns(pol, expected_reward, rng):
    """Assert the fitted policy beats a uniform-random policy and mostly agrees with the oracle."""
    Xq = rng.normal(size=(4000, NFEATURES))
    oracle = expected_reward(Xq).argmax(axis=1)
    chosen = np.asarray(pol.predict(Xq)).astype(int)

    r_pol = _sample_continuous_reward(rng, expected_reward, Xq, chosen)
    r_rand = _sample_continuous_reward(
        rng, expected_reward, Xq, rng.integers(0, NCHOICES, size=Xq.shape[0])
    )
    agreement = float((chosen == oracle).mean())

    assert r_pol.mean() > r_rand.mean()
    assert agreement > 0.6  # observed ~0.92 across seeds; 0.6 is a wide deterministic margin


def test_linucb_continuous_rewards():
    pol, expected_reward, rng = _fit_on_random_logging_policy(
        lambda: LinUCB(nchoices=NCHOICES, beta_prior=None, smoothing=None, random_state=SEED),
        SEED,
    )
    _assert_learns(pol, expected_reward, rng)


@pytest.mark.parametrize("base", [LinearRegression, Ridge])
def test_bootstrapped_ucb_regressor_base_continuous_rewards(base):
    pol, expected_reward, rng = _fit_on_random_logging_policy(
        lambda: BootstrappedUCB(
            base(), nchoices=NCHOICES, beta_prior=None, smoothing=None, random_state=SEED
        ),
        SEED,
    )
    _assert_learns(pol, expected_reward, rng)


def test_linucb_partial_fit_continuous_rewards():
    """Streaming updates with continuous rewards run without error and yield valid choices."""
    rng = np.random.default_rng(SEED)
    pol = LinUCB(nchoices=NCHOICES, beta_prior=None, smoothing=None, random_state=SEED)
    for _ in range(10):
        Xb = rng.normal(size=(100, NFEATURES))
        ab = rng.integers(0, NCHOICES, size=100)
        rb = np.clip(rng.random(100), 0.0, 1.0)  # continuous rewards in [0, 1]
        pol.partial_fit(Xb, ab, rb)
    pred = np.asarray(pol.predict(rng.normal(size=(20, NFEATURES)))).astype(int)
    assert pred.shape == (20,)
    assert ((pred >= 0) & (pred < NCHOICES)).all()


def test_partitioned_ucb_rejects_continuous_rewards():
    """Boundary: PartitionedUCB uses sklearn's DecisionTreeClassifier internally, which
    rejects continuous targets. This documents why tree-based partitioned policies are
    excluded from the continuous-reward set."""
    rng = np.random.default_rng(SEED)
    pol = PartitionedUCB(nchoices=NCHOICES, beta_prior=None, smoothing=None, random_state=SEED)
    X = rng.normal(size=(400, NFEATURES))
    a = rng.integers(0, NCHOICES, size=400)
    # Continuous rewards with many distinct values per arm so the tree actually attempts a fit
    # (the internal model skips fitting when an arm sees <= 1 unique reward value).
    r = np.clip(rng.random(400), 0.0, 1.0)
    with pytest.raises(ValueError, match="continuous"):
        pol.fit(X, a, r)
