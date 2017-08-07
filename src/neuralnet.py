from src.core import sigmoid
import numpy

class NeuralNet:
    layers = []

    def __init__(self):
        pass

    def add_layer(self, size):
        """

        :param size: The size of the new layer
        :return:
        """
        self.layers.append(size)

    def forward_propagate(self, X, theta):
        """

        :param X: A numpy array of input features. This does not include the bias feature.
        :param theta: A list of thetas for each layer, including a theta for the bias node.
        :return:
        """
        activations = []
        m = numpy.shape(X)[0]
        X_with_bias = numpy.concatenate((numpy.ones([m, 1]), X), axis=1)

        # There is no need to run the activation function on the input layer; its just the raw features
        previous_activation = X_with_bias
        for i in range(1, len(self.layers)):
            z = previous_activation.dot(theta[i - 1].transpose())
            activation = sigmoid(z)
            activations.append(activation)
            previous_activation = activation
            numpy.concatenate((numpy.ones([m, 1]), previous_activation), axis=1)

        return activations


    def backward_propagate(self):
        pass