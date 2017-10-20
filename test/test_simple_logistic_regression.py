import unittest
import numpy
from numpy import *
import random
import src.core as core
import src.utils as utils
import src.gradient_check as gradient_check
import src.data_loader as data_loader

class TestSimpleLogisticRegression(unittest.TestCase):

    def setUp(self):
        random.seed(1)
        numpy.random.seed(1)
        X, Y = data_loader.get_sample_data()
        self.X = X
        self.Y = Y
        W, b = core.initialize_weights(3, 1, 1.0)
        self.W = W
        self.b = b

    def test_simple_logistic_regression(self):
        diff = gradient_check.gradient_check_simple_logistic(self.X, self.Y, self.W, self.b, epsilon=1e-7)
        self.assertLess(diff, 5e-5)

if __name__ == '__main__':
    unittest.main()
