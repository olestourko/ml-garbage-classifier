import unittest
import numpy
from numpy import *
from src.main import sigmoid


class TestSigmoid(unittest.TestCase):
    def test_sigmoid(self):
        X = array([[1, 1, 1], [1, 1, 1]])
        theta = array([[1, 1, 1]]).transpose()
        z = X.dot(theta)
        hypothesis = sigmoid(z)

        expected = array([[0.95257], [0.95257]])
        numpy.testing.assert_almost_equal(expected, hypothesis, 5)

if __name__ == '__main__':
    unittest.main()