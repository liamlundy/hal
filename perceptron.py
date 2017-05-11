import numpy as np

from utils.arrays import normalize


def predict(input_var, weights, bias):
    # assert input_var[:-1].shape == weights.shape
    if np.dot(input_var, weights) + bias > 0:
        return 1
    else:
        return 0


def update_weights(input_var, target, weights, bias, rate):
    predicted = predict(input_var, weights, bias)
    error = target - predicted

    new_bias = bias + rate * error
    new_weights = weights + rate * error * input_var

    return new_weights, new_bias


def train_once(input_vars, targets, weights, bias, rate):
    for input_var, target in zip(input_vars, targets):
        weights, bias = update_weights(input_var, target, weights, bias, rate)
    return weights, bias


def train_n_times(input_vars, targets, rate, n):
    weights = np.zeros(input_vars.shape[1])
    bias = 0
    for i in range(n):
        weights, bias = train_once(input_vars, targets, weights, bias, rate)
    return weights, bias


# TODO: THis is temporary
def did_it_work(input_vars):
    weights, bias = train_n_times(normalize(input_vars[:, :-1]), input_vars[:, -1], 0.1, 5)
    count = 0
    for input_var, target in zip(normalize(input_vars[:, :-1]), input_vars[:, -1]):
        if predict(input_var, weights, bias) == target:
            count += 1
    print("Correct {:.2%} of the time".format(count/input_vars.shape[0]))


def test_with_long_data():
    training_data = np.loadtxt('data/longdata.dat')
    weights, bias = train_n_times(normalize(training_data[:, :-1]), training_data[:, -1], 0.01, 5)
    count = 0
    print('Weights: {}, bias: {}'.format(weights, bias))

    testing_data = normalize(np.loadtxt('data/shortdata.dat')[:, :-1])
    testing_target = np.loadtxt('data/shortdata.dat')[:, -1]
    for input_var, target in zip(testing_data, testing_target):
        if predict(input_var, weights, bias) == target:
            count += 1
    print("Correct {:.2%} of the time".format(count/testing_data.shape[0]))


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

# print(test_with_long_data())
#
# print(did_it_work(data_set))

# print(train_once(normalize(data_set), np.array([0, 0]), 0, .1))
