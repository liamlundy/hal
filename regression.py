import numpy as np
import matplotlib.pyplot as plt


def compute_grad(X, y):
    """
    Takes an array of inputs and array of outputs and determines which 
    coefficients most accurately model this set.
    :param X: Nd-array of inputs with all cells in col 1 equaling 1
    :param y: Nd-array of outputs for training
    :return: Nd-array of coefficients
    """
    return (np.linalg.inv(X.T.dot(X))).dot(X.T).dot(y)


def create_training_set(input_data):
    """
    Takes an array of input data, adds col of 1's, col of x values, and
    col of x values squared.
    :param input_data: Nd-array of inputs and output data
    :return: The formatted set of inputs
    """
    ones = np.ones(input_data.shape[0])
    return np.column_stack((ones, input_data[:, 0], input_data[:, 0] ** 2))


if __name__ == "__main__":
    # Load the data
    data = np.loadtxt('parabolic.dat')
    # Format the input data
    training_set = create_training_set(data)
    # Strip the out data
    y_values = np.column_stack((data[:, 1])).T

    # Compute the coefficients
    w = compute_grad(training_set, y_values)

    # Write the equation to a file
    with open('equation.dat', 'w') as result:
        result.write('f(x) = {:.3f} + {:.3f}*x + {:.3f}*x^2'.format(w[0][0], w[1][0], w[2][0]))

    # Get the x data points
    lin = data[:, 0]
    # plot the data
    plt.plot(data[:, 0], data[:, 1], 'r.')
    # plot the fitted line
    plt.plot(lin, w[0][0] + w[1][0] * lin + w[2][0] * (lin ** 2), 'b-')
    plt.show()
