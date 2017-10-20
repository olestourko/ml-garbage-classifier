import unittest
import numpy
from numpy import *
import random
import src.neuralnet as neuralnet
import src.utils as utils
import src.gradient_check as gradient_check
import src.data_loader as data_loader

class TestBackProp(unittest.TestCase):

    def setUp(self):
        random.seed(1)
        numpy.random.seed(1)
        X, Y = data_loader.get_sample_data()
        self.X = X
        self.Y = Y
        self.nn = neuralnet.NeuralNet([3, 3, 1])
        self.weights = self.nn.initialize_weights()
        print(self.weights)

    def test_back_prop(self):
        # nn, X, nn_weights, Y
        diff = gradient_check.gradient_check_nn(self.nn, self.X, self.weights, self.Y, epsilon=1e-7)
        self.assertLess(diff, 5e-5)

if __name__ == '__main__':
    unittest.main()
