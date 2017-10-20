from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import src.core as core
import src.neuralnet as neuralnet
import src.gradient_check as gradient_check
import src.utils as utils
import random, numpy
from src import data_loader

random.seed(1)
numpy.random.seed(1)

""" Try flattening weights for simple logistic regression """
W, b = core.initialize_weights(3, 1, 1.0)
flattened_weights = utils.weights_to_vector(W, b)

W2, b2 = utils.vector_to_weights(flattened_weights, 3, 1)

random.seed(1)
numpy.random.seed(1)

""" Try flattening weights for a neural network """
nn = neuralnet.NeuralNet([3, 3, 1])
nn_weights = nn.initialize_weights()
flattened_nn_weights = utils.nn_weights_to_vector(nn_weights)
nn_weights_2 = utils.vector_to_nn_weights(flattened_nn_weights, nn.layers)

""" Fetch the sample data """
X, Y = data_loader.get_sample_data()

""" Try gradient checking on simple logistic regression """
diff = gradient_check.gradient_check_simple_logistic(X, Y, W, b)

""" Try gradient checking on a neural network """
nn_diff = gradient_check.gradient_check_nn(nn, X, nn_weights, Y)