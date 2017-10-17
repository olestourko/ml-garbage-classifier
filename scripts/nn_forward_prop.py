from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from src.neuralnet import *
import numpy
from numpy import shape, ones

# Load data
raw_data = numpy.loadtxt("../resources/ex2data1.txt", delimiter=',')
X = raw_data[:, 0:2].T
# Add an extra feature
ratio_feature = X[0, :] / X[1, :]
X = numpy.vstack((X, ratio_feature))

Y = raw_data[:, -1:].T
m = shape(X)[1] # number of training examples
n = shape(X)[0] # number of input features

# Initialize a neural network
nn = NeuralNet([n, 1])

# Randomly initialize weights
weights = nn.initialize_weights()

# Forward prop on one of the training examples
activations, _ = nn.forward_propagate(X[:, 0:1], weights)

print("Number of activations: {}".format(len(activations)))
for i, a in enumerate(activations):
    print("Shape of activation {}: {}".format(i, a.shape))