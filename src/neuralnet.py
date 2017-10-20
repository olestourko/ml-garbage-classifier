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
            print(i)
            if i == len(self.layers) - 1:
                dz = activations[i] - Y
                dW = (1.0 / m) * dz.dot(activations[i - 1].T)
                db = (1.0 / m) * numpy.sum(dz, axis=1, keepdims=True)
            else:
                print(weights[i])

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