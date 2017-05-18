import numpy as np


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


def train(input_vars, targets, rate, n):
    weights = np.zeros(input_vars.shape[1])
    bias = 0
    for i in range(n):
        weights, bias = train_once(input_vars, targets, weights, bias, rate)
    return weights, bias
