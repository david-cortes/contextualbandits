import numpy as np
import pytest

from contextualbandits.evaluation import evaluateDoublyRobust


@pytest.fixture
def data():
    X = np.array([[1., 0.], [0., 1.], [1., 1.]])
    pred = np.array([0, 1, 1])
    a = np.array([0, 0, 1])
    r = np.array([1, 0, 0])
    p = np.array([0.5, 0.25, 0.8])
    estimates = np.array([[0.2, 0.2], [0.6, 0.7], [0.4, 0.4]])
    return pred, X, a, r, p, estimates


@pytest.mark.parametrize("handle_invalid", [False, True])
def test_doubly_robust_repeated_evaluation(data, handle_invalid):
    first = evaluateDoublyRobust(*data, handle_invalid=handle_invalid)
    second = evaluateDoublyRobust(*data, handle_invalid=handle_invalid)
    assert second == pytest.approx(first)


@pytest.mark.parametrize("handle_invalid", [False, True])
def test_doubly_robust_preserves_reward_estimates(data, handle_invalid):
    if handle_invalid:
        data[-1][:] = [[1., 1.], [0., 1.], [0., 0.]]
    original = data[-1].copy()
    evaluateDoublyRobust(*data, handle_invalid=handle_invalid)
    np.testing.assert_array_equal(data[-1], original)


@pytest.mark.parametrize("handle_invalid", [False, True])
def test_doubly_robust_readonly_reward_estimates(data, handle_invalid):
    expected = evaluateDoublyRobust(*data, handle_invalid=handle_invalid)
    data[-1].setflags(write=False)
    result = evaluateDoublyRobust(*data, handle_invalid=handle_invalid)
    assert result == pytest.approx(expected)
