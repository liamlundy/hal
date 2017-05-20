import gzip
import pickle

import numpy as np

from hal.utils.arrays import multi_class_to_matrix


def total_net_input(input_vars, weights, bias):
    assert True  # Check dims
    return np.dot(weights, input_vars) + bias


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
    def __init__(self, ):
        self.learning_rate = None

        self.input_layer = None
        self.output_layer = None
        self.hidden_layers = []

        self.weights = []

        self.labeler = Labeler()

    def initialize_network(self, num_inputs, num_outputs, hidden_layers_sizes, learning_rate):
        self.learning_rate = learning_rate

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
        # Batch of size 1

        def create_partials_matrix(activations, error):
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

    def train(self, training_set, training_targets, n):
        for _ in range(n):
            for inputs, targets in zip(training_set, training_targets):
                self.feed_forward(inputs)
                self.back_prop(targets)
                self.gradient_descent()

    def fit(self, inputs, targets, hidden_layer_sizes, learning_rate=0.01, iterations=10):
        # Number of input columns

        # is it multi class or binary?
        # TODO: binary how many outputs?
        unique_target_values = np.unique(targets)
        if len(unique_target_values) > 2:

            # TODO: move this
            # self.type = 'multiclass'
            targets = multi_class_to_matrix(targets)
            self.labeler.set_labels(unique_target_values)
        else:
            raise NotImplementedError("Only works for multiclass as of now.")

        self.initialize_network(inputs.shape[1], len(unique_target_values), hidden_layer_sizes, learning_rate)

        # TODO: How many iterations?
        self.train(inputs, targets, iterations)

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
# class MLP:
#
#     def __init__(self, num_hidden_nodes=3, learning_rate=0.01):
#         self.type = None
#         self.learning_rate = learning_rate
#
#         self.num_input_nodes = None
#         self.num_output_nodes = None
#         self.num_hidden_nodes = num_hidden_nodes
#
#         self.weights1 = None
#         self.weights2 = None
#
#         self.bias1 = None
#         self.bias2 = None
#
#     def __repr__(self):
#         info = 'Activation function:\tNone\nLearning rate:\t{}\nFirst set of weights:\n{}\nSecond set of ' \
#                'weights:\n{}\nFirst layer bias:\t{}\nSecond layer bias:\t{}'.format(self.learning_rate,
#                                                                                     self.weights1, self.weights2,
#                                                                                     self.bias1, self.bias2)
#         return info
#
#     def update_weight(self, input_vars, target, weights1, weights2, bias1, bias2):
#         hidden_nodes = sigmoid(total_net_input(input_vars, weights1, bias1))
#
#         output_nodes = sigmoid(total_net_input(hidden_nodes, weights2, bias2))
#
#         delta = MLP.delta_output(output_nodes, target)
#         new_weights2 = weights2 - self.learning_rate*np.column_stack(delta*row for row in hidden_nodes)
#
#         total = np.array([sum(delta * col) for col in weights2.T])
#
#         new_weights1 = np.copy(weights1)
#         for row in range(weights1.shape[0]):
#             new_weights1[:, row] = weights1[:, row] - self.learning_rate * total * hidden_nodes * \
#                                                       (1 - hidden_nodes) * input_vars[row]
#
#         return new_weights1, new_weights2
#
#     def train_once(self, input_vars, targets):
#         for input_var, target in zip(input_vars, targets):
#             self.weights1, self.weights2 = self.update_weight(
#                 input_var, target, self.weights1, self.weights2, self.bias1, self.bias2
#             )
#
#     def fit(self, input_vars, targets, **kwargs):
#         # Number of input columns
#         self.num_input_nodes = input_vars.shape[1]
#
#         # is it multi class or binary?
#         # TODO: binary how many outputs?
#         unique_target_values = np.unique(targets)
#         if len(unique_target_values) > 2:
#             self.type = 'multiclass'
#             self.num_output_nodes = len(unique_target_values)
#             targets = multi_class_to_matrix(targets)
#         else:
#             raise NotImplementedError("Only works for multiclass as of now.")
#
#         # TODO: stop hard coding
#         # TODO: multiple hidden layers
#         self.num_hidden_nodes = 3
#
#         # Initialize value for the weights
#         interval = 4*np.sqrt(6 / (self.num_hidden_nodes + self.num_input_nodes))
#         self.weights1 = np.random.uniform(-interval, interval, size=(self.num_hidden_nodes, self.num_input_nodes))
#         interval = 4*np.sqrt(6 / (self.num_output_nodes + self.num_hidden_nodes))
#         self.weights2 = np.random.uniform(-interval, interval, size=(self.num_output_nodes, self.num_hidden_nodes))
#
#         # Initialize biases
#         self.bias1 = np.zeros(self.num_hidden_nodes)
#         self.bias2 = np.zeros(self.num_output_nodes)
#
#         # TODO: How many iterations?
#         self.train(input_vars, targets, 100)
#
#     def train(self, input_vars, targets, n):
#         # values = targets.unique()
#         # if sum(values) > 2:
#
#         multi_class_to_matrix(targets[:, -1])
#
#         for i in range(n):
#             self.train_once(input_vars, targets)
#
#     def predict_once(self, input_var):
#         hidden_nodes = sigmoid(total_net_input(input_var, self.weights1, self.bias1))
#         output_nodes = sigmoid(total_net_input(hidden_nodes, self.weights2, self.bias2))
#
#         return output_nodes
#
#     def predict(self, input_vars):
#         return np.apply_along_axis(self.predict_once, 1, input_vars)
#
#     @staticmethod
#     def total_error(output, target):
#         return sum(0.5*(target - output)**2)
#
#     @staticmethod
#     def delta_output(output, target):
#         return -(target - output)*output*(1 - output)


# mlp = MLP(3, 3, 1)
# mlp.train(np.array([[.05, .1, .05]]), np.array([2]), 10000)
#
# print(mlp.predict(np.array([[0.05, 0.1, .05]])))
#
# print('\n' + repr(mlp))


if __name__ == "__main__":
    with gzip.open('/home/liam/Downloads/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    net = Network()

    net.fit(train_set[0], train_set[1], (400, ), iterations=5)

    predicted = net.predict(test_set[0])
    print(predicted)
    print(np.average(predicted == test_set[1]))
    print(net)
