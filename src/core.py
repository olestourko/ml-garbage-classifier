from math import e
import numpy
from numpy import shape, array
from numpy import log as log


def sigmoid(z):
    """
    Runs a hypothesis through the sigmoid function.
    :param z: A scalar or matrix hypothesis
    :return: the hypothesis run through the sigmoid function
    """
    return 1.0 / (1.0 + (e ** -z))

def sigmoid_derivative(sigmoid_output):
    """
    :param z: A scalar or matrix hypothesis, should be the output of a sigmoid call
    :return:
    """
    return sigmoid_output * (1.0 - sigmoid_output)

""" New functions below for updates from Deep Learning course"""
"""
General matrix dimensions
X: n-1 x m
Y: 1 x m
W: n-1 x n (layer input size x layer output size)
b: n x 1 (layer output size)

z: n x m
a: n x m 
"""

def initialize_weights(input_size, layer_size, epsilon=1.0):
    """
    Matrix dimensions:
    https://www.coursera.org/learn/neural-networks-deep-learning/lecture/Y20qP/explanation-for-vectorized-implementation
    https://www.coursera.org/learn/neural-networks-deep-learning/lecture/tyAGh/computing-a-neural-networks-output

    :param input_size: The number of nodes (or input features) in the previous layer
    :param layer_size: The number of nodes in this layer
    :param epsilon: The size of the range
    :return: (W, b)
    """

    W = numpy.random.uniform(-epsilon / 2.0, epsilon / 2.0, (input_size, layer_size))
    b = numpy.zeros((layer_size, 1))
    return W, b

def calculate_z(X, W, b):
    """
    https://www.coursera.org/learn/neural-networks-deep-learning/lecture/moUlO/vectorizing-logistic-regression

    :param X:
    :param W:
    :param b:
    :return: z (n x m matrix)
    """

    # Do some sanity checks on the matrix sizes
    n_minus_one = X.shape[0]
    m = X.shape[1]
    assert W.shape[0] == n_minus_one
    n = W.shape[1]
    assert b.shape[0] == n
    assert b.shape[1] == 1

    z = W.T.dot(X) + b
    return z

def logistic_cost_function(X, a, Y):
    """
    :param X: n-1 x m
    :param a: The final activations (n x m matrix)
    :param Y: Expected outputs (1 x m matrix)
    :return: (cost, derivative wrt weights, derivative wrt bias)
    """

    m = a.shape[1]
    # Do some sanity checks on the matrix sizes
    assert Y.shape[0] == 1
    assert Y.shape[1] == m

    # https://www.coursera.org/learn/machine-learning/lecture/1XG8G/cost-function
    j = (1.0 / m) * numpy.sum(
        -Y * log(a)
        -
        (1.0 - Y) * log(1.0 - a)
    )

    dz = a - Y
    dW = (1.0 / m) * dz.dot(X.T)
    db = (1.0 / m) * numpy.sum(dz, axis=1, keepdims=True)

    return j, dW, db

def minimize(activation_cost_function, X, Y, W, b, learning_rate, iterations):
    """
    :param activation_cost_function: Function which returns the cost and gradients (expects X, Y, W, b)
    Note: This function computes both the activation and the gradients
    :param cost_function: Function which returns the cost and gradients (expects X, a, Y)
    :param X: Input features (n-1 x m matrix)
    :param Y: Expected outputs (1 x m matrix)
    :param W: Randomly initialized weights (n-1 x n matrix)
    :param b: Bias term (1 x n matrix)
    :param learning_rate:
    :param iterations:
    :return: costs, W, b
    """

    # Do some sanity checks on the matrix sizes
    # n_minus_one = X.shape[0]
    # m = X.shape[1]
    # assert Y.shape[0] == 1
    # assert Y.shape[1] == m
    # assert W.shape[0] == n_minus_one
    # n = W.shape[1]
    # assert b.shape[0] == n
    # assert b.shape[1] == 1

    costs = []
    W = W.copy()
    b = b.copy()

    for i in range(0, iterations):
        j, dW, db = activation_cost_function(X, Y, W, b)
        """
        The notes use dW.T for simple logistic regression, but dW for neural networks.
        """
        W -= (learning_rate * dW)
        b -= (learning_rate * db)
        costs.append(j)

    return costs, W, b

def minimize_with_momentum(activation_cost_function, X, Y, W, b, learning_rate, momentum_weight, iterations):
    """
    https://www.coursera.org/learn/deep-neural-network/lecture/y0m1f/gradient-descent-with-momentum
    https://devblogs.nvidia.com/parallelforall/deep-learning-nutshell-history-training/

    :param activation_cost_function: Function which returns the cost and gradients (expects X, W, b)
    Note: This function computes both the activation and the gradients
    :param X: Input features (n-1 x m matrix)
    :param Y: Expected outputs (1 x m matrix)
    :param W: Randomly initialized weights (n-1 x n matrix)
    :param b: Bias term (1 x n matrix)
    :param learning_rate:
    :param momentum_weight: The lower, the lesser the effect of momentum. Default it to ~0.95
    :param iterations:
    :return: costs, W, b
    """

    costs = []
    W = W.copy()
    b = b.copy()
    # momentum terms
    vdW = numpy.zeros(W.shape)
    vdb = numpy.zeros(b.shape)

    for i in range(0, iterations):
        j, dW, db = activation_cost_function(X, Y, W, b)
        # These are the momentum terms (uses exponentially weighted averages)
        if i == 0:
            vdW = dW
            vdb = db
        else:
            vdW = (momentum_weight * vdW) + ((1.0 - momentum_weight) * dW)
            vdb = (momentum_weight * vdb) + ((1.0 - momentum_weight) * db)

        W -= (learning_rate * vdW)
        b -= (learning_rate * vdb)
        costs.append(j)

    return costs, W, b

def predict(activation_function, X, W, b):
    return numpy.round(activation_function(calculate_z(X, W, b)))