import matplotlib.pyplot as plt
import numpy as np

from hal.utils.arrays import multi_class_to_matrix


def sigmoid(weighted_input):
    """
    Sigmoid activation function
    :param weighted_input: The weighted input from the previous layer
    :type weighted_input: np.array
    :return: The input array with the activation function applied
    :rtype: np.array
    """
    return 1.0 / (1.0 + np.exp(-weighted_input))


def sigmoid_prime(weighted_input):
    """
    The derivative of the sigmoid activation function
    :param weighted_input: The weighted input from the previous layer
    :type weighted_input: np.array
    :return: The input array with the derivative activation function applied
    :rtype: np.array
    """
    return sigmoid(weighted_input)*(1 - sigmoid(weighted_input))


class Layer:
    """
    Class to represent a generate layer
    """
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weighted_inputs = np.empty([num_neurons])
        self.activations = np.empty([num_neurons])
        self.biases = np.zeros([num_neurons])
        self.error = np.zeros([num_neurons])

    def compute_weighted_inputs(self, weights, activations):
        """
        Compute the weighted inputs to this layer using the weights linking this layer
        and the previous layer and the activations from the previous layer.
        :param weights: Weights linking this layer and the previous layer
        :type weights: np.array
        :param activations: The activations from the previous layer
        :type activations: np.array
        :return: The weighted inputs for this layer
        :rtype: np.array
        """
        self.weighted_inputs = np.dot(weights, activations) + self.biases

    def compute_activations(self):
        """
        Computes the activations for this layer from the weighted input
        :return: The activations for this layer
        :rtype: np.array
        """
        self.activations = sigmoid(self.weighted_inputs)

    def compute_error(self, weights, next_layer_errors):
        """
        Computes the error for this layer.
        :param weights: Weights connecting this layer to the next
        :type weights: np.array
        :param next_layer_errors: Error from the next layer
        :type next_layer_errors: np.array
        :return: The error for this layer
        :rtype: np.array
        """
        self.error = np.multiply(np.dot(weights.T, next_layer_errors), sigmoid_prime(self.weighted_inputs))


class Network:
    """
    Class to construct a neural network and use it to train a model.
    """
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
        """
        Initialize the network with the given parameters
        :param num_inputs: Dimension of the input vector
        :type num_inputs: int
        :param num_outputs: Dimension of the output vector
        :type num_outputs: int
        :param hidden_layers_sizes: The number of neurons to be contained in each hidden layer.
        :type hidden_layers_sizes: tuple
        :param learning_rate: The initial learning rate of the network
        :type learning_rate: int
        :param batch_size: Number of inputs to train at once
        :type batch_size: int
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.input_layer = Layer(num_inputs)
        self.output_layer = Layer(num_outputs)
        for num_neurons in hidden_layers_sizes:
            self.hidden_layers.append(Layer(num_neurons))

        def create_weight_matrix(num_input_nodes, num_output_nodes):
            """
            Create initial weight matrix, using random values in the interval best 
            suited for the sigmoid function.
            :param num_input_nodes: Number of input nodes
            :type num_input_nodes: int
            :param num_output_nodes: Number of output nodes
            :type num_output_nodes: int
            :return: Initial weight matrix of correct dimensions
            :rtype: np.array
            """
            interval = 4 * np.sqrt(6 / (num_input_nodes + num_output_nodes))
            return np.random.uniform(-interval, interval, size=(num_output_nodes, num_input_nodes))

        # Initial an array of weights to connect each layer
        self.weights = [
            create_weight_matrix(num_inputs, self.hidden_layers[0].num_neurons)
        ]
        for index in range(len(self.hidden_layers) - 1):
            self.weights.append(create_weight_matrix(self.hidden_layers[index].num_neurons,
                                                     self.hidden_layers[index + 1].num_neurons))
        self.weights.append(create_weight_matrix(self.hidden_layers[-1].num_neurons, self.output_layer.num_neurons))

    def feed_forward(self, inputs):
        """
        Feed the training example through the network, computing weighted inputs 
        and activations for each layer.
        :param inputs: The training set
        :type inputs: np.array
        """
        # Compute for input layer
        self.input_layer.weighted_inputs = inputs
        self.input_layer.compute_activations()

        # Set first layer activations in temp var to be fed into next layer
        prev_activations = self.input_layer.activations
        for index in range(len(self.hidden_layers)):
            # Compute for each hidden layer
            self.hidden_layers[index].compute_weighted_inputs(self.weights[index], prev_activations)
            self.hidden_layers[index].compute_activations()
            prev_activations = self.hidden_layers[index].activations

        self.output_layer.compute_weighted_inputs(self.weights[-1], prev_activations)
        self.output_layer.compute_activations()

    def back_prop(self, targets):
        """
        Back propagate the error through each layer
        :param targets: The ideal value for this training example
        :type targets: np.array
        """
        # Output error
        self.output_layer.error = np.multiply(self.output_layer.activations - targets,
                                              sigmoid_prime(self.output_layer.weighted_inputs))

        # Set the output error as the error to be fed into te previous layers
        next_layer_error = self.output_layer.error
        for index in range(len(self.hidden_layers) - 1, -1, -1):
            # Feed error into each hidden layer
            self.hidden_layers[index].compute_error(self.weights[index + 1], next_layer_error)
            next_layer_error = self.hidden_layers[index].error

        self.input_layer.compute_error(self.weights[0], next_layer_error)

    def gradient_descent(self):
        """
        Change the weights and biases according to the back-propagated error for each weight and bias.
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

        # Start with the last layer
        prev_activations = self.input_layer.activations
        for index in range(len(self.hidden_layers)):
            # Update weights
            self.weights[index] -= self.learning_rate*create_partials_matrix(prev_activations,
                                                                             self.hidden_layers[index].error)
            # Update biases
            self.hidden_layers[index].biases -= self.learning_rate*self.hidden_layers[index].error
            prev_activations = self.hidden_layers[index].activations
        self.weights[-1] -= self.learning_rate*create_partials_matrix(prev_activations, self.output_layer.error)
        self.output_layer.biases -= self.learning_rate*self.output_layer.error

    def train(self, training_set, training_targets, n, batch_size):
        """
        Train a set of training examples n times using batch training.
        :param training_set: The set of training examples.
        :type training_set: np.array
        :param training_targets: The targets for each training example
        :type training_targets: np.array
        :param n: The number of times to train  the network on the training set
        :type n: int
        :param batch_size: The number of examples to constitute a batch
        :type batch_size: int
        """
        # TODO: Online training
        err = []
        for _ in range(n):
            error_sum = 0
            for inputs, targets in zip(training_set, training_targets):
                # Run the model
                self.feed_forward(inputs)
                self.back_prop(targets)
                self.gradient_descent()
                error_sum += np.sum(self.output_layer.error**2)  # Record total error
            # Update the learning rate depending on how the error changes
            if len(err) > 0:
                if error_sum < err[-1]:
                    self.learning_rate *= 0.95
                else:
                    self.learning_rate *= 0.5
            err.append(error_sum)
        if self.verbose:
            # Show the gradient descent curve
            plt.plot(err)
            plt.show()

    def fit(self, inputs, targets, hidden_layer_sizes, learning_rate=0.01, iterations=10, batch_size=32, verbose=False):
        """
        Initiate and train the network.
        :param inputs: The set of training examples.
        :type inputs: np.array
        :param targets: The ideal values for each training example.
        :type targets: np.array
        :param hidden_layer_sizes: The number of neurons to be in each hidden layer
        :type hidden_layer_sizes: tuple
        :param learning_rate: The initial rate at which the network learns
        :type learning_rate: int
        :param iterations: The number of times to train the network on the training set
        :type iterations: int
        :param verbose: Display graphs and log?
        :type verbose: bool
        """
        self.verbose = verbose
        # TODO: binary mode
        # Number of outputs
        unique_target_values = np.unique(targets)
        if len(unique_target_values) > 2:
            self.type = 'multiclass'
            # Create target binary matrix
            targets = multi_class_to_matrix(targets)
            # Save labels for multiclass
            self.labeler.set_labels(unique_target_values)
        else:
            raise NotImplementedError("Only works for multiclass as of now.")

        # Set up the netowrk with teh given pararmeters
        self.initialize_network(inputs.shape[1], len(unique_target_values), hidden_layer_sizes, learning_rate,
                                batch_size)

        # TODO: Early stopping
        self.train(inputs, targets, iterations, self.batch_size)

    def predict(self, testing_set):
        """
        Predict the output for a set of examples with the current weights.
        :param testing_set: The set of inputs to predict
        :type testing_set: np.array
        :return: An array of labels
        :rtype: np.array
        """
        outputs = []
        for inputs in testing_set:
            self.feed_forward(inputs)
            outputs.append(self.output_layer.activations)
        return self.labeler.to_labels(np.array(outputs))


class Labeler:
    """
    Save and transform labels for multiclass data
    """
    def __init__(self):
        self.labels = None

    def set_labels(self, labels):
        self.labels = labels

    def to_labels(self, predicted):
        max_indexes = np.argmax(predicted, axis=1)
        return np.array([self.labels[index] for index in max_indexes])
