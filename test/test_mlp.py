import numpy as np
from numpy.testing import assert_almost_equal

from hal.mlp import sigmoid, Network


def test_sigmoid():
    assert_almost_equal(sigmoid(0.3775), 0.593269992)


def test_weights():
    mlp = MLP(2, 2, 2)
    new_weights1, new_weights2 = mlp.update_weight(np.array([0.05, 0.1]), np.array([0.01, 0.99]),
                                                   np.array([[0.15, 0.2], [.25, .3]]),
                                                   np.array([[0.4, 0.45], [.5, .55]]), .35, .6)
    assert_almost_equal(new_weights1, np.array([[0.149780716, 0.19956143], [0.24975114, 0.29950229]]))
    assert_almost_equal(new_weights2, np.array([[0.35891648, 0.408666186], [0.511301270, 0.561370121]]))


def test_train():
    mlp = MLP(2, 2, 2)
    mlp.weights1 = np.array([[0.15, 0.2], [.25, .3]])
    mlp.weights2 = np.array([[0.4, 0.45], [.5, .55]])
    mlp.bias1 = 0.35
    mlp.bias2 = 0.6
    mlp.train(np.array([[0.05, 0.1]]), np.array([[0.01, 0.99]]), 10000)
    assert_almost_equal(mlp.predict(np.array([[0.05, 0.1]])), np.array([[0.015912196, 0.984065734]]), 5)
