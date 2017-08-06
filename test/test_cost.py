import unittest
import numpy
from numpy import *
from src.core import logisticCostFunction


class TestCost(unittest.TestCase):

    def setUp(self):
        self.X = array([[1, 1, 1], [1, 1, 1]])
        self.y = array([[1], [1]])
        self.theta = array([[1, 1, 1]]).transpose()

    def test_cost(self):
        j, gradients = logisticCostFunction(self.X, self.y, self.theta)
        expected = 0.048587
        numpy.testing.assert_almost_equal(expected, j, 5)

    def test_gradients(self):
        j, gradients = logisticCostFunction(self.X, self.y, self.theta)
        expected = array([[-0.047426, -0.047426, -0.047426]])
        numpy.testing.assert_almost_equal(expected, gradients, 5)

if __name__ == '__main__':
    unittest.main()
