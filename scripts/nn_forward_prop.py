from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from src.core import *
from src.neuralnet import *
import numpy
from numpy import shape, ones

# Load data
raw_data = numpy.loadtxt("../resources/ex2data2.txt", delimiter=',')
X = raw_data[:, 0:2]
Y = raw_data[:, -1:]
m = shape(X)[0] # number of training examples
n = shape(X)[1] # number of features

bias_features = numpy.ones([m, 1])
X_with_bias = numpy.concatenate((bias_features, X), axis=1)
theta_with_bias = [
    ones([3, n+1]), # number of nodes, number of features (including a theta for the bias node)
    ones([6, 3]), # number of nodes, number of features. The bias node is added automatically after the first layer.
    ones([1, 6])
]

# Randomly initialize thetas
random_theta = randomly_initiate_theta(theta_with_bias, 1)

# Initialize a neural network
nn = NeuralNet(n)
nn.add_layer(3) # input layer
nn.add_layer(6) # hidden layer layer
nn.add_layer(1) # output layer
activations = nn.forward_propagate(X, random_theta)

print("Number of activations: {}".format(len(activations)))
for i, activation in enumerate(activations):
    print("Shape of activation {}: {}".format(i, numpy.shape(activation)))