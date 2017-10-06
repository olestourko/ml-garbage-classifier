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
    return 1.0 / (1.0 + (e ** -z))

def sigmoid_derivative(sigmoid_output):
    """
    :param z: A scalar or matrix hypothesis, should be the output of a sigmoid call
    :return:
    """
    return sigmoid_output * (1.0 - sigmoid_output)

def predict(X, theta):
    return numpy.round(sigmoid(X.dot(theta)))

def logisticCostFunction(X, Y, rlambda, theta):
    """
    Computes cost and gradients for a data set
    :param X: i by j matrix of training features
    :param y: i by 1 matrix of training results
    :param rlambda: the regularization constant
    :param theta: 1 by j matrix of thetas
    :return: the total cost for the particular theta and the gradients
    """
    # the number of training examples
    m = shape(X)[0]
    hypothesis = sigmoid(X.dot(theta))

    # https://www.coursera.org/learn/machine-learning/lecture/1XG8G/cost-function
    j = (1.0 / m) * sum(
        -Y * log(hypothesis)
        -
        (1.0 - Y) * log(1.0 - hypothesis)
    )
    # Add regularization
    j += sum((rlambda / m) * theta)

    # Get gradients
    gradients = (1.0 / m) * (hypothesis - Y).transpose().dot(X)
    theta_for_regularization = theta[:]
    gradients += (rlambda / m) * theta_for_regularization.transpose()

    return j, gradients

def minimize(cost_function, X, y, initial_theta, alpha, rlambda, iterations):
    """

    :param cost_function: the cost function to minimize. It returns the cost and gradients.
    :param X: i by j matrix of training features
    :param y: i by 1 matrix of training results
    :param initial_theta: 1 by j matrix of thetas
    :param alpha: learning rate (smaller == slower, but it can't be too large because gradient descent will break
    :param rlambda: the regularization constant
    :param iterations: the number of gradient descent iteration
    :return: thetas found through gradient descent and cost at each iteration
    """

    initial_alpha = alpha
    costs = []
    theta = initial_theta[:]

    n_increased_alpha = 0;
    n_decreased_alpha = 0;

    for i in range(0, iterations):
        cost, gradients = cost_function(X, y, rlambda, theta)

        # Dynamically adjust the learning rate
        if i > 0:
            if costs[i - 1] > cost:
                # Gradient descent is converging
                alpha += initial_alpha
                n_increased_alpha += 1
            else:
                # Gradient descent is diverging
                alpha = max([initial_alpha, alpha / 2.0])
                # cost, gradients = cost_function(X, y, rlambda, theta) # Recompute cost and gradients
                n_decreased_alpha += 1

        theta -= alpha * gradients.transpose() #subtract the derivative of the cost function from the current theta(s)
        costs.append(cost)

    # print(n_increased_alpha)
    # print(n_decreased_alpha)
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

def randomly_intiate_theta_rolled(n, epsilon):
    """

    :param n: The length of the rolled theta list
    :param epsilon: The range in which to generate thetas. theta(i, j) in epsilon < 0 < epsilon
    :return: The randomly initialized thetas
    """
    return [numpy.random.uniform(-epsilon, epsilon) for i in range(0, n)]