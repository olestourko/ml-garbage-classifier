import unittest
import numpy
from numpy import *
from src.main import cost


class TestCost(unittest.TestCase):
    def test_cost(self):
        X = array([[1, 1, 1], [1, 1, 1]])
        y = array([[1], [1]])
        theta = array([[1, 1, 1]]).transpose()
        j = cost(X, y, theta)
        expected = 0.048587
        numpy.testing.assert_almost_equal(expected, j, 5)

if __name__ == '__main__':
    unittest.main()
