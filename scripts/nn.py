from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

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
theta_with_bias = [ones([1, n+1])]

# Initialize a neural network
nn = NeuralNet()
nn.add_layer(3) # input layer
nn.add_layer(1) # output layer
activations = nn.forward_propagate(X, theta_with_bias)
print(activations)
