import unittest
import numpy
from numpy import *
from src.core import logistic_cost_function, sigmoid, calculate_z


class TestCost(unittest.TestCase):

    def setUp(self):
        self.X = array([[1, 1, 1], [1, 1, 1]]).T
        self.Y = array([[1, 1]])
        self.W = array([[1, 1, 1]]).T
        self.b = array([[0]])

    def test_cost(self):
        z = calculate_z(self.X, self.W, self.b)
        a = sigmoid(z)
        j, dW, db = logistic_cost_function(self.X, a, self.Y)
        expected = 0.048587
        numpy.testing.assert_almost_equal(j, expected, 5)

if __name__ == '__main__':
    unittest.main()
