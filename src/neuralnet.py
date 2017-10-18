import src.core as core
from src.core import sigmoid, sigmoid_derivative
import numpy

class NeuralNet:
    def __init__(self, layers):
        """

        :param layers: A list of layer sizes (Must have at least two elements: the input layer size and the output layer of 1)
        """

        assert(len(layers)) >= 2
        self.layers = layers

    def predict(self, X, weights):
        activations, _ = self.forward_propagate(X, weights)
        last_activation = activations[-1].flatten()
        return numpy.round(last_activation)
        # return last_activation

    def forward_propagate(self, X, weights):
        """

        :param X: Input features (n x m matrix)
        :param weights: A list of (W, b) tuples, one for each layer
        :return: List of activations, list of zs
        """

        activations = [X] # The inputs are the 0th activation
        zs = []

        for i in range(1, len(self.layers)): # Skip the 0th layer, since these are just the inputs
            W = weights[i - 1]["W"]
            b = weights[i - 1]["b"]
            z = core.calculate_z(activations[i - 1], W, b)
            a = core.sigmoid(z)

            zs.append(z)
            activations.append(a)

        return activations, zs

    def backward_propagate(self, X, activations, zs, Y, weights):
        """
        https://www.coursera.org/learn/neural-networks-deep-learning/lecture/Wh8NI/gradient-descent-for-neural-networks

        :param X: A numpy array of input features. This does not include the bias feature.
        :param Y: A numpy array of training results.
        :param activations:
        :param zs:
        :param weights: A list of (W, b) tuples, one for each layer
        :return:
        """

        assert(len(weights) == len(self.layers) - 1)

        m = X.shape[1]

        dzs = {}
        gradients = []
        # The 1st layer is ignored (its just the input layer)
        for i in reversed(range(1, len(self.layers))):
            if i == len(self.layers) - 1:
                dz = activations[i] - Y
                dW = (1.0 / m) * dz.dot(activations[i - 1].T)
                db = (1.0 / m) * numpy.sum(dz, axis=1, keepdims=True)
            else:
                prev_dz = dzs[i + 1]
                z = zs[i]
                W = weights[i]["W"]
                b = weights[i]["b"]
                dz = W.dot(prev_dz) * core.sigmoid_derivative(z)
                dW = (1.0 / m) * dz.dot(activations[i - 1].T)
                db = (1.0 / m) * numpy.sum(dz, axis=1, keepdims=True)

            dzs[i] = dz
            gradients.append({
                "W": dW,
                "b": db
            })

        return list(reversed(gradients))

    def initialize_weights(self):
        """

        :return: A list of W, b dicts (one tuple for each layer)
        """
        weights = []
        for i in range(0, len(self.layers) - 1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            W, b = core.initialize_weights(layer, next_layer)
            weights.append({
                "W": W,
                "b": b
            })

        return weights

def roll_weights(nn, weights):
    """

    :param nn: NeuralNetwork instance
    :param W: List of W, tuples
    :return: * x 1 matrix of W's, * x 1 matrix of b's
    """
    # 1. Determine rolled weight array sizes
    W_size = 0
    b_size = 0
    for i in range(1, len(nn.layers)):
        W_size += nn.layers[i - 1] * nn.layers[i]
        b_size += nn.layers[i]

    W_matrix = numpy.zeros((1, W_size))
    b_matrix = numpy.zeros((1, b_size))

    # 2. Roll the W's and b's
    last_W_index = 0
    last_b_index = 0
    for i in range(1, len(nn.layers)):
        W_size = nn.layers[i - 1] * nn.layers[i]
        b_size = nn.layers[i]
        W_matrix[0, last_W_index:last_W_index + W_size] = weights[i - 1]["W"].flatten()
        b_matrix[0, last_b_index:last_b_index + b_size] = weights[i - 1]["b"].flatten()
        last_W_index = W_size
        last_b_index = b_size

    return W_matrix, b_matrix

def unroll_weights(nn, W, b):
    """

    :param nn: NeuralNetwork instance
    :param W: * x 1 matrix
    :param b: * x 1 matrix
    :return: List of W, b tuples
    """
    weights = []

    last_W_index = 0
    last_b_index = 0
    for i in range(1, len(nn.layers)):
        W_size = nn.layers[i - 1] * nn.layers[i]
        b_size = nn.layers[i]
        weights.append({
            "W": W[0, last_W_index:last_W_index + W_size].reshape(nn.layers[i - 1], nn.layers[i]),
            "b": b[0, last_b_index:last_b_index + b_size].reshape((nn.layers[i], 1))
        })
        last_W_index = W_size
        last_b_index = b_size

    return weights