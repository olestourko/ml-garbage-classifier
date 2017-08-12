from src.core import sigmoid
import numpy

class NeuralNet:
    def __init__(self, input_features):
        """

        :param input_features: The number of input features, excluding the bias feature.
        """
        self.layers = []
        self.input_features = input_features
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
        for i in range(0, len(self.layers)):
            if i == 0:
                activations.append(X_with_bias)
            else:
                z = activations[i - 1].dot(theta[i - 1].transpose())
                activation = sigmoid(z)
                numpy.concatenate((numpy.ones([m, 1]), activation), axis=1)
                activations.append(activation)

        return activations


    def backward_propagate(self, X, Y, theta, lambda_reg):
        """

        :param X: A numpy array of input features. This does not include the bias feature.
        :param Y: A numpy array of training results.
        :param theta: A flat list of thetas, including bias nodes
        :param lambda_reg: A scalar which controls the regularization weight in the cost function
        :return: the total cost for the particular theta and the gradients (unrolled)
        """
        m = numpy.shape(X)[0]
        theta = self.roll_thetas(theta)
        activations = self.forward_propagate(X, theta)

        # Compute the cost of the thetas (excluding regularization)
        hypothesis = activations[-1]
        j = -(1.0 / m) * numpy.sum((Y * numpy.log(hypothesis)) + ((1 - Y) * numpy.log(1 - hypothesis)))

        errors = {}
        accumulators = {}
        gradients = {}
        for i, layer_size in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1:
                # Compute the first error term at the output layer
                errors[i] = activations[i] - Y
            else:
                # Computer error terms
                this_layers_theta = theta[i]
                sigmoid_derivative = ((activations[i] * (1 - activations[i])))
                errors[i] = (errors[i + 1].dot(this_layers_theta)) * sigmoid_derivative
                accumulators[i] = activations[i].transpose().dot(errors[i+1])

                # Regularization
                regularization_term = lambda_reg * this_layers_theta
                regularization_term[:, 0] = 0 # Don't regularize the bias terms

                gradients[i] = ((1.0 / m) * accumulators[i]) + regularization_term.transpose()

        # Unroll gradients into a single vector
        unrolled_gradients = []
        for key, value in gradients.items():
            unrolled_gradients.extend(numpy.ravel(value).tolist())

        return j, unrolled_gradients

    def roll_thetas(self, theta):
        """

        :param theta: A flat list of thetas
        :return: The same thetas as a list of numpy arrays
        """
        t = 0  # Total number of thetas unrolled
        rolled_thetas = []
        non_input_layers = self.layers[1:] # Skip the input layer
        for i, n_nodes in enumerate(non_input_layers):
            if i == 0:
                l = n_nodes * (self.input_features + 1)  # Number of thetas in this layer
                rolled = numpy.array(theta[t:l]).reshape([n_nodes, self.input_features + 1])
                rolled_thetas.append(rolled)
                t += l
            else:
                l = n_nodes * non_input_layers[i - 1]  # Number of thetas in this layer
                # print(numpy.array(theta[t:t + l]))
                # print(n_nodes, self.layers[i - 1])
                rolled = numpy.array(theta[t:t + l]).reshape([n_nodes, non_input_layers[i - 1]])
                rolled_thetas.append(rolled)
                t += l

        return rolled_thetas