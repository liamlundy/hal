from numpy import *


def total_net_input(input_vars, weights, bias):
    assert True  # Check dims
    return dot(weights, input_vars) + bias


def activation(net_input):
    return 1 / (1 + exp(-net_input))


class MLP:

    def __init__(self, num_input_nodes, num_output_nodes, num_hidden_nodes, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        self.num_hidden_nodes = num_hidden_nodes

        self.weights1 = zeros((num_hidden_nodes, num_input_nodes))
        self.weights2 = zeros((num_output_nodes, num_hidden_nodes))

        self.bias1 = 0
        self.bias2 = 0

    def update_weight(self, input_vars, target, weights1, weights2, bias1, bias2):
        hidden_nodes = activation(total_net_input(input_vars, weights1, bias1))

        output_nodes = activation(total_net_input(hidden_nodes, weights2, bias2))

        delta = MLP.delta_output(output_nodes, target)
        new_weights2 = weights2 - self.learning_rate*column_stack([delta*hidden_nodes[0], delta*hidden_nodes[1]])

        total = array([sum(delta*weights2[:, 0]), sum(delta*weights2[:, 1])])

        new_weights1 = copy(weights1)
        for row in range(weights1.shape[0]):
            new_weights1[:, row] = weights1[:, row] - self.learning_rate * total * hidden_nodes * \
                                                      (1 - hidden_nodes) * input_vars[row]

        return new_weights1, new_weights2
    
    def train(self, input_vars, targets):
        for input_var, target in zip(input_vars, targets):
            self.weights1, self.weights2 = self.update_weight(
                input_vars, target, self.weights1, self.weights2, self.bias1, self.bias2
            )

    @staticmethod
    def total_error(output, target):
        return sum(0.5*(target - output)**2)

    @staticmethod
    def delta_output(output, target):
        return -(target - output)*output*(1 - output)


# def delta_rule_w5(output, target):
    # -(target, output)*output*(1 - output)*output


mlp = MLP(2, 1, 2)
res = mlp.update_weight(array([.05, .1]), array([0.01, 0.99]), array([[.15, .2], [.25, .3]]), array([[.4, .45], [.5, .55]]), .35, .6)

err = mlp.total_error(res, array([0.01, 0.99]))
print(res)

# d = mlp.delta_output(res, array([0.01, 0.99]))
# print(d)

# gradient(mlp.)

#
# net = total_net_input(array([[.15, .2], [.25, .3]]), array([.05, .1]), .35)
#
# print(net)
#
# hidden_out = activation(net)
#
# print(hidden_out)
#
# out = total_net_input(array([[.4, .45], [.5, .55]]), hidden_out, .6)
#
# print(activation(out))
