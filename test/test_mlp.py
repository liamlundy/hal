from numpy.testing import assert_almost_equal

from mlp import activation


def test_activation():
    assert_almost_equal(activation(0.3775), 0.593269992)
