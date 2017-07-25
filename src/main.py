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
    Computes cost and gradients for a data set
    :param X: i by j matrix of training features
    :param y: i by 1 matrix of training results
    :param theta: 1 by j matrix of thetas
    :return: the total cost for the particular theta and the gradients
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

def minimize(X, y, initial_theta, alpha, iterations):
    """

    :param X: i by j matrix of training features
    :param y: i by 1 matrix of training results
    :param initial_theta: 1 by j matrix of thetas
    :param alpha: learning rate (smaller == slower, but it can't be too large because gradient descent will break
    :param iterations: the number of gradient descent iteration
    :return: thetas found through gradient descent
    """
    pass

if __name__ == "__main__":
    X = array([[1, 1, 1], [1, 1, 1]])
    y = array([[1], [1]])
    theta = array([[1, 1, 1]]).transpose()

    j, gradients = cost(X, y, theta)