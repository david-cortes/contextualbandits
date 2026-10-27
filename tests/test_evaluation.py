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
@pytest.mark.parametrize("readonly", [False, True], ids=["writable", "readonly"])
def test_doubly_robust_repeated_evaluation(data, handle_invalid, readonly):
    if handle_invalid:
        data[-1][:] = [[1., 1.], [0., 1.], [0., 0.]]
    for array in data:
        array.setflags(write=not readonly)
    first = evaluateDoublyRobust(*data, handle_invalid=handle_invalid, random_state=1)
    second = evaluateDoublyRobust(*data, handle_invalid=handle_invalid, random_state=1)
    assert second == first


@pytest.mark.parametrize("handle_invalid", [False, True])
@pytest.mark.parametrize("readonly", [False, True], ids=["writable", "readonly"])
def test_doubly_robust_preserves_inputs(data, handle_invalid, readonly):
    if handle_invalid:
        data[-1][:] = [[1., 1.], [0., 1.], [0., 0.]]
    for array in data:
        array.setflags(write=not readonly)
    originals = [array.copy() for array in data]
    evaluateDoublyRobust(*data, handle_invalid=handle_invalid)
    for array, original in zip(data, originals):
        np.testing.assert_array_equal(array, original)
