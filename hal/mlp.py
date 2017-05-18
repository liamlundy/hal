import numpy as np


def total_net_input(input_vars, weights, bias):
    assert True  # Check dims
    return np.dot(weights, input_vars) + bias


def activation(net_input):
    return 1 / (1 + np.exp(-net_input))


class MLP:

    def __init__(self, num_input_nodes, num_output_nodes, num_hidden_nodes, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        self.num_hidden_nodes = num_hidden_nodes

        self.weights1 = np.zeros((num_hidden_nodes, num_input_nodes))
        self.weights2 = np.zeros((num_output_nodes, num_hidden_nodes))

        self.bias1 = 0
        self.bias2 = 0

    def __repr__(self):
        info = 'Activation function:\tNone\nLearning rate:\t{}\nFirst set of weights:\n{}\nSecond set of ' \
               'weights:\n{}\nFirst layer bias:\t{}\nSecond layer bias:\t{}'.format(self.learning_rate,
                                                                                    self.weights1, self.weights2,
                                                                                    self.bias1, self.bias2)
        return info

    def update_weight(self, input_vars, target, weights1, weights2, bias1, bias2):
        hidden_nodes = activation(total_net_input(input_vars, weights1, bias1))

        output_nodes = activation(total_net_input(hidden_nodes, weights2, bias2))

        delta = MLP.delta_output(output_nodes, target)
        # new_weights2 = weights2 - self.learning_rate*np.column_stack([delta*hidden_nodes[0], delta*hidden_nodes[1]])
        new_weights2 = weights2 - self.learning_rate*np.column_stack(delta*row for row in hidden_nodes)

        # total = np.array([sum(delta*weights2[:, 0]), sum(delta*weights2[:, 1])])
        total = np.array([sum(delta * col) for col in weights2.T])
        # np.apply_along_axis(sum, 0, weights2 * delta)

        new_weights1 = np.copy(weights1)
        for row in range(weights1.shape[0]):
            new_weights1[:, row] = weights1[:, row] - self.learning_rate * total * hidden_nodes * \
                                                      (1 - hidden_nodes) * input_vars[row]

        return new_weights1, new_weights2
    
    def train_once(self, input_vars, targets):
        for input_var, target in zip(input_vars, targets):
            self.weights1, self.weights2 = self.update_weight(
                input_var, target, self.weights1, self.weights2, self.bias1, self.bias2
            )

    def train(self, input_vars, targets, n):
        for i in range(n):
            self.train_once(input_vars, targets)

    def predict_once(self, input_var):
        hidden_nodes = activation(total_net_input(input_var, self.weights1, self.bias1))
        output_nodes = activation(total_net_input(hidden_nodes, self.weights2, self.bias2))

        return output_nodes

    def predict(self, input_vars):
        return np.apply_along_axis(self.predict_once, 1, input_vars)

    @staticmethod
    def total_error(output, target):
        return sum(0.5*(target - output)**2)

    @staticmethod
    def delta_output(output, target):
        return -(target - output)*output*(1 - output)


mlp = MLP(3, 3, 1)
mlp.train(np.array([[.05, .1, .05]]), np.array([2]), 10000)

print(mlp.predict(np.array([[0.05, 0.1, .05]])))

print('\n' + repr(mlp))
