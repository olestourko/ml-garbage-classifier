import unittest
import numpy
from numpy import *
from src.core import logistic_cost_function, sigmoid


class TestCost(unittest.TestCase):

    def setUp(self):
        self.X = array([[1, 1, 1], [1, 1, 1]]).T
        self.Y = array([[1], [1]])
        self.W = array([[1, 1, 1]])
        self.b = 0

    def test_cost(self):
        j, gradients = logistic_cost_function(sigmoid, self.X, self.Y, self.W, self.b) # Set rlmabda param to 0 to ignore regularization
        expected = 0.048587
        numpy.testing.assert_almost_equal(expected, j, 5)

    def test_gradients(self):
        j, gradients = logistic_cost_function(sigmoid, self.X, self.Y, self.W, self.b) # Set rlmabda param to 0 to ignore regularization
        expected = array([[-0.047426, -0.047426, -0.047426]])
        numpy.testing.assert_almost_equal(expected, gradients, 5)

if __name__ == '__main__':
    unittest.main()
