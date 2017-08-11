from src.core import sigmoid
import numpy

class NeuralNet:
    def __init__(self):
        self.layers = []
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
        :return: The activations for every layer
        """
        activations = []
        m = numpy.shape(X)[0]
        X_with_bias = numpy.concatenate((numpy.ones([m, 1]), X), axis=1)

        # There is no need to run the activation function on the input layer; its just the raw features
        previous_activation = X_with_bias
        for i in range(1, len(self.layers) + 1):
            z = previous_activation.dot(theta[i - 1].transpose())
            activation = sigmoid(z)
            activations.append(activation)
            previous_activation = activation
            numpy.concatenate((numpy.ones([m, 1]), previous_activation), axis=1)

        return activations


    def backward_propagate(self, X, Y, theta):
        """

        :param X: A numpy array of input features. This does not include the bias feature.
        :param Y: A numpy array of training results.
        :param theta: A list of thetas for each layer, including a theta for the bias node.
        :return:
        """
        activations = self.forward_propagate(X, theta)

        errors = {}
        for i, layer_size in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1:
                errors[i] = activations[i] - Y
                # Compute the first error term
                pass
            else:
                this_layers_theta = theta[i + 1]
                # print(i)
                # print(errors[i + 1].dot(this_layers_theta))
                sigmoid_derivative = ((activations[i] * (1 - activations[i])))
                errors[i] = (errors[i + 1].dot(this_layers_theta)) * sigmoid_derivative
