import numpy as np
from numpy.testing import assert_array_almost_equal

from utils.arrays import normalize


def test_normalize():
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

    normalized = np.array([[1.00000000e-01, 6.40277609e-01],
                           [1.50000000e-01, 3.83577812e-04],
                           [2.00000000e-01, 2.21570880e-01],
                           [2.50000000e-01, 9.44005135e-01],
                           [3.00000000e-01, 3.82285417e-02],
                           [3.50000000e-01, 6.24202897e-01],
                           [4.00000000e-01, 8.11137889e-01],
                           [4.50000000e-01, 1.52473398e-01],
                           [5.00000000e-01, 4.49639998e-01],
                           [5.50000000e-01, 7.86589627e-01],
                           [6.00000000e-01, 8.69038247e-01],
                           [6.50000000e-01, 7.89038192e-01],
                           [7.00000000e-01, 4.54151897e-01],
                           [7.50000000e-01, 7.72837974e-01],
                           [8.00000000e-01, 2.05645421e-01],
                           [8.50000000e-01, 8.89642170e-01]])
    assert_array_almost_equal(normalize(data_set[:, :-1]), normalized)