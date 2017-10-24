from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import src.core as core
import src.neuralnet as neuralnet
import src.gradient_check as gradient_check
import random, numpy
from src import data_loader

random.seed(10)
numpy.random.seed(10)

"""
Fetch the sample data

Normalize the sample data too
This is necessary because even with well-initiated weights, z grows large,
which causes sigmoid(z) to round off to 0 or 1. This breaks the calculations
when numerically computing gradients (can't divide by 0, can't take the logarithm of 0)

"""
X, Y = data_loader.get_sample_data(normalized=True)

X = X[:, 0:1]
Y = Y[:, 0:1]

""" Try gradient checking on simple logistic regression """
for i in range (0, 10):
    W, b = core.initialize_weights(3, 1)
    diff = gradient_check.gradient_check_simple_logistic(X, Y, W, b)

random.seed(1)
numpy.random.seed(1)

""" Try gradient checking on a neural network """
nn = neuralnet.NeuralNet([3, 1, 1])
for i in range (0, 1):
    nn_weights = nn.initialize_weights()
    nn_diff = gradient_check.gradient_check_nn(nn, X, nn_weights, Y)
    print(nn_diff)