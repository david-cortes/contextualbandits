import numpy as np
import pytest

from contextualbandits.evaluation import evaluateDoublyRobust, evaluateNCIS


def _data():
    X = np.array([[1., 0.], [0., 1.], [1., 1.]])
    pred = np.array([0, 1, 1])
    a = np.array([0, 0, 1])
    r = np.array([1, 0, 0])
    p = np.array([0.5, 0.25, 0.8])
    estimates = np.array([[0.2, 0.2], [0.6, 0.7], [0.4, 0.4]])
    return pred, X, a, r, p, estimates


@pytest.mark.parametrize("handle_invalid", [False, True])
def test_doubly_robust_repeated_evaluation(handle_invalid):
    args = _data()
    first = evaluateDoublyRobust(*args, handle_invalid=handle_invalid)
    second = evaluateDoublyRobust(*args, handle_invalid=handle_invalid)
    assert first == pytest.approx((1.8 + 0.6 - 0.1) / 3)
    assert second == pytest.approx(first)


@pytest.mark.parametrize("handle_invalid", [False, True])
def test_doubly_robust_preserves_reward_estimates(handle_invalid):
    args = list(_data())
    if handle_invalid:
        args[-1] = np.array([[1., 1.], [0., 1.], [0., 0.]])
    original = args[-1].copy()
    evaluateDoublyRobust(*args, handle_invalid=handle_invalid)
    np.testing.assert_array_equal(args[-1], original)


@pytest.mark.parametrize("handle_invalid", [False, True])
def test_doubly_robust_readonly_reward_estimates(handle_invalid):
    args = _data()
    args[-1].setflags(write=False)
    result = evaluateDoublyRobust(*args, handle_invalid=handle_invalid)
    assert result == pytest.approx((1.8 + 0.6 - 0.1) / 3)


@pytest.mark.parametrize("column_inputs", [False, True])
@pytest.mark.parametrize("c", [None, 0.5])
@pytest.mark.parametrize("pmin", [None, 0.3])
def test_doubly_robust_propensity_controls(column_inputs, c, pmin):
    pred, X, a, r, p, estimates = _data()
    adjusted = p.copy()
    if c is not None:
        adjusted *= c
    if pmin is not None:
        adjusted = np.maximum(adjusted, pmin)
    expected = np.mean([
        estimates[i, 0] + (r[i] - estimates[i, 1]) / adjusted[i]
        if pred[i] == a[i] else estimates[i, 0]
        for i in range(len(r))
    ])
    if column_inputs:
        pred, a, r, p = [v.reshape(-1, 1) for v in [pred, a, r, p]]
    result = evaluateDoublyRobust(pred, X, a, r, p, estimates,
                                 handle_invalid=False, c=c, pmin=pmin)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("column_inputs", [False, True])
def test_ncis_clips_small_weights_and_discards_large_weights(column_inputs):
    est = np.array([0.05, 0.5, 0.9])
    p = np.array([0.5, 0.25, 0.1])
    r = np.array([1, 0, 1])
    if column_inputs:
        est, r, p = [v.reshape(-1, 1) for v in [est, r, p]]
    assert evaluateNCIS(est, r, p, cmin=0.2, cmax=3.) == pytest.approx(0.2 / 2.2)
