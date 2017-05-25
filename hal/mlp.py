import gzip
import pickle
from sklearn import datasets

import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import StandardScaler

from hal.utils.arrays import multi_class_to_matrix


def sigmoid(weighted_input):
    return 1.0 / (1.0 + np.exp(-weighted_input))


def sigmoid_prime(weighted_input):
    return sigmoid(weighted_input)*(1 - sigmoid(weighted_input))


class Layer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weighted_inputs = np.empty([num_neurons])
        self.activations = np.empty([num_neurons])
        self.biases = np.zeros([num_neurons])
        self.error = np.zeros([num_neurons])

    def compute_weighted_inputs(self, weights, activations):
        self.weighted_inputs = np.dot(weights, activations) + self.biases

    def compute_activations(self):
        self.activations = sigmoid(self.weighted_inputs)

    def compute_error(self, weights, next_layer_errors):
        self.error = np.multiply(np.dot(weights.T, next_layer_errors), sigmoid_prime(self.weighted_inputs))


class Network:
    def __init__(self):
        self.learning_rate = None

        self.input_layer = None
        self.output_layer = None
        self.hidden_layers = []

        self.weights = []

        self.batch_size = None
        self.verbose = False
        self.type = None

        self.labeler = Labeler()

    def initialize_network(self, num_inputs, num_outputs, hidden_layers_sizes, learning_rate, batch_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.input_layer = Layer(num_inputs)
        self.output_layer = Layer(num_outputs)
        for num_neurons in hidden_layers_sizes:
            self.hidden_layers.append(Layer(num_neurons))

        def create_weight_matrix(num_input_nodes, num_output_nodes):
            interval = 4 * np.sqrt(6 / (num_input_nodes + num_output_nodes))
            return np.random.uniform(-interval, interval, size=(num_output_nodes, num_input_nodes))

        self.weights = [
            create_weight_matrix(num_inputs, self.hidden_layers[0].num_neurons)
        ]
        for index in range(len(self.hidden_layers) - 1):
            self.weights.append(create_weight_matrix(self.hidden_layers[index].num_neurons,
                                                     self.hidden_layers[index + 1].num_neurons))
        self.weights.append(create_weight_matrix(self.hidden_layers[-1].num_neurons, self.output_layer.num_neurons))

    def feed_forward(self, inputs):
        self.input_layer.weighted_inputs = inputs
        self.input_layer.compute_activations()
        prev_activations = self.input_layer.activations

        for index in range(len(self.hidden_layers)):
            self.hidden_layers[index].compute_weighted_inputs(self.weights[index], prev_activations)
            self.hidden_layers[index].compute_activations()
            prev_activations = self.hidden_layers[index].activations

        self.output_layer.compute_weighted_inputs(self.weights[-1], prev_activations)
        self.output_layer.compute_activations()

    def back_prop(self, targets):
        self.output_layer.error = np.multiply(self.output_layer.activations - targets,
                                              sigmoid_prime(self.output_layer.weighted_inputs))
        next_layer_error = self.output_layer.error

        for index in range(len(self.hidden_layers) - 1, -1, -1):
            self.hidden_layers[index].compute_error(self.weights[index + 1], next_layer_error)
            next_layer_error = self.hidden_layers[index].error

        self.input_layer.compute_error(self.weights[0], next_layer_error)

    def gradient_descent(self):
        """
        
        :return: 
        :rtype: 
        """

        # act_in * delta_out for each neuron pair
        def create_partials_matrix(activations, error):
            """
            Creates matrices to multiply each pair of neurons that span a weight.
            :param activations: the activations from the previous layer
            :type activations: np.array
            :param error: the errors from the next layer
            :type error: np.array
            :return: the partial derivative of the error for each weight
            :rtype: np.array
            """
            return np.multiply(np.repeat(activations[None, :], len(error), axis=0),
                               np.repeat(error[:, None], len(activations), axis=1))

        prev_activations = self.input_layer.activations
        for index in range(len(self.hidden_layers)):
            np.repeat(self.input_layer.activations[None, :], len(self.hidden_layers[index].error), axis=0)
            self.weights[index] -= self.learning_rate*create_partials_matrix(prev_activations,
                                                                             self.hidden_layers[index].error)
            self.hidden_layers[index].biases -= self.learning_rate*self.hidden_layers[index].error
            prev_activations = self.hidden_layers[index].activations
        self.weights[-1] -= self.learning_rate*create_partials_matrix(prev_activations, self.output_layer.error)
        self.output_layer.biases -= self.learning_rate*self.output_layer.error

    def train(self, training_set, training_targets, n, batch_size):
        # TODO: Batch size
        err = []
        for _ in range(n):
            error_sum = 0
            for inputs, targets in zip(training_set, training_targets):
                self.feed_forward(inputs)
                self.back_prop(targets)
                self.gradient_descent()
                error_sum += np.sum(self.output_layer.error**2)
            if len(err) > 0:
                if error_sum < err[-1]:
                    self.learning_rate *= 0.95
                else:
                    self.learning_rate *= 0.5
            err.append(error_sum)
        if self.verbose:
            plt.plot(err)
            plt.show()

    def fit(self, inputs, targets, hidden_layer_sizes, learning_rate=0.01, iterations=10, batch_size=32, verbose=False):
        self.verbose = verbose
        # is it multi class or binary?
        # TODO: binary how many outputs?
        unique_target_values = np.unique(targets)
        if len(unique_target_values) > 2:

            # TODO: move this
            self.type = 'multiclass'
            targets = multi_class_to_matrix(targets)
            self.labeler.set_labels(unique_target_values)
        else:
            raise NotImplementedError("Only works for multiclass as of now.")

        self.initialize_network(inputs.shape[1], len(unique_target_values), hidden_layer_sizes, learning_rate,
                                batch_size)

        # TODO: How many iterations?
        self.train(inputs, targets, iterations, self.batch_size)

    def predict(self, testing_set):
        outputs = []
        for inputs in testing_set:
            self.feed_forward(inputs)
            outputs.append(self.output_layer.activations)
        return self.labeler.to_labels(np.array(outputs))


class Labeler:
    def __init__(self):
        self.labels = None

    def set_labels(self, labels):
        self.labels = labels

    def to_labels(self, predicted):
        max_indexes = np.argmax(predicted, axis=1)
        return np.array([self.labels[index] for index in max_indexes])
