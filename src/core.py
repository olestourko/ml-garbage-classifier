from math import e
import numpy
from numpy import shape, array
from numpy import log as log


def sigmoid(z):
    """
    Runs a hypothesis through the sigmoid function.
    :param z: A scalar or matrix hypothesis
    :return: the hypothesis run through the sigmoid function
    """
    return 1 / (1 + (e ** -z))

def predict(X, theta):
    return numpy.round(sigmoid(X.dot(theta)))

def logisticCostFunction(X, y, theta):
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

def minimize(cost_function, X, y, initial_theta, alpha, iterations):
    """

    :param cost_function: the cost function to minimize. It returns the cost and gradients.
    :param X: i by j matrix of training features
    :param y: i by 1 matrix of training results
    :param initial_theta: 1 by j matrix of thetas
    :param alpha: learning rate (smaller == slower, but it can't be too large because gradient descent will break
    :param iterations: the number of gradient descent iteration
    :return: thetas found through gradient descent and cost at each iteration
    """

    costs = []
    theta = initial_theta[:]

    for i in range(0, iterations):
        cost, gradients = cost_function(X, y, theta)
        costs.append(cost)
        theta -= alpha * gradients.transpose() #subtract the derivative of the cost function from the current theta(s)

    return theta, costs

def randomly_initiate_theta(theta, epsilon):
    """

    :param theta: A numpy array or list of numpy arrays where the thetas will go
    :param epsilon: The range in which to generate thetas. theta(i, j) in epsilon < 0 < epsilon
    :return: the randomly initialized thetas
    """

    if isinstance(theta, list):
        random_theta = theta[:]
    else:
        random_theta = [numpy.copy(theta)]

    for i, t in enumerate(random_theta):
        random_theta[i] = numpy.random.uniform(-epsilon, epsilon, shape(t))

    return random_theta

if __name__ == "__main__":
    X = array([[1, 1, 1], [1, 1, 1]])
    y = array([[1], [1]])
    theta = array([[1, 1, 1]]).transpose()

    j, gradients = logisticCostFunction(X, y, theta)