from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import src.core as core
import src.neuralnet as neuralnet
import src.gradient_check as gradient_check
import random, numpy

random.seed(1)
numpy.random.seed(1)

""" Try flattening weights for simple logistic regression """
W, b = core.initialize_weights(3, 1)
flattened_weights = gradient_check.weights_to_vector(W, b)
W2, b2 = gradient_check.vector_to_weights(flattened_weights, 3, 1)

""" Try flattening weights for a neural network """
nn = neuralnet.NeuralNet([3, 3, 1])
nn_weights = nn.initialize_weights()
flattened_nn_weights = gradient_check.nn_weights_to_vector(nn_weights)
nn_weights_2 = gradient_check.vector_to_nn_weights(flattened_nn_weights, [3, 3, 1])