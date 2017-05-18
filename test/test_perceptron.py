import numpy as np
from numpy.testing import assert_almost_equal

from hal.perceptron import predict, update_weights, train_once
from hal.utils.arrays import normalize

data_set = np.array([[10, 64.0277609352, 0],
                     [15, 0.0383577812151, 0],
                     [20, 22.15708796, 0],
                     [25, 94.4005135336, 1],
                     [30, 3.8228541672, 0],
                     [35, 62.4202896763, 1],
                     [40, 81.1137889117, 0],
                     [45, 15.2473398102, 0],
                     [50, 44.9639997899, 1],
                     [55, 78.6589626868, 1],
                     [60, 86.9038246994, 1],
                     [65, 78.9038191825, 1],
                     [70, 45.4151896937, 1],
                     [75, 77.2837974455, 1],
                     [80, 20.5645421131, 1],
                     [85, 88.9642169694, 1]])


def test_predict():
    assert predict(np.array([0.5, 1]), np.array([-2, 0.5]), 0.8) == 1


def test_update_weights_weights():
    assert_almost_equal(update_weights(np.array([0.5, 1]), 0, np.array([-2, 0.5]), 0.8, 0.1)[0],
                        np.array([-2.05, 0.4]))


def test_update_weights_bias():
    assert update_weights(np.array([0.5, 1]), 0, np.array([-2, 0.5]), 0.8, 0.1)[1] == np.float64(0.70000000000000007)


def test_train_once_weights():
    assert_almost_equal(train_once(normalize(data_set[:, :-1]), data_set[:, -1], np.array([0, 0]), 0, 0.01)[0],
                        np.array([-0.00542326, 0.00133366]))


def test_train_once_bias():
    assert train_once(normalize(data_set[:, :-1]), data_set[:, -1], np.array([0, 0]), 0, 0.01)[1] == 0.01
