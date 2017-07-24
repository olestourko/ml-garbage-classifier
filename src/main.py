from math import *
from numpy import *
from numpy import log as log


def sigmoid(z):
    """
    Runs a hypothesis through the sigmoid function.
    :param z: A scalar or matrix hypothesis
    :return: the hypothesis run through the sigmoid function
    """
    return 1 / (1 + (e ** -z))


def cost(X, y, theta):
    """

    :param X: i by j matrix of training features
    :param y: i by 1 matrix of training results
    :param theta: 1 by j matrix of thetas
    :return: the total cost for the particular theta
    """
    # the number of training examples
    m = shape(X)[0]
    hypothesis = sigmoid(X.dot(theta))

    # https://www.coursera.org/learn/machine-learning/lecture/1XG8G/cost-function
    j = (1.0 / m) * sum(
        - y * log(hypothesis)
        -
        (1.0 - y) * log(1.0 - hypothesis)
    )
    gradients = (1.0 / m) * ((hypothesis.transpose() - y.transpose()).dot(X))

    return j, gradients


if __name__ == "__main__":
    X = array([[1, 1, 1], [1, 1, 1]])
    y = array([[1], [1]])
    theta = array([[1, 1, 1]]).transpose()

    j, gradients = cost(X, y, theta)