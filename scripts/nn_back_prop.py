from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import src.core as core
import src.neuralnet as neuralnet
import numpy
from numpy import shape, ones

# Load data
raw_data = numpy.loadtxt("../resources/ex2data1.txt", delimiter=',')
X = raw_data[:, 0:2].T
# Add an extra feature
ratio_feature = X[0, :] / X[1, :]
X = numpy.vstack((X, ratio_feature))

Y = raw_data[:, -1:].T
m = shape(X)[1] # number of training examples
n = shape(X)[0] # number of input features

# Initialize a neural network
# nn = neuralnet.NeuralNet([n, 1]) # This is the same as simple logistic regression
nn = neuralnet.NeuralNet([n, 3, 1])
# Randomly initialize weights
weights = nn.initialize_weights()

learning_rate = 0.001
iterations = 50000

def activation_cost_function(X, Y, W, b):
    weights = neuralnet.unroll_weights(nn, W, b)
    activations, zs = nn.forward_propagate(X, weights)
    gradients = nn.backward_propagate(X, activations, zs, Y, weights)
    dW_rolled, db_rolled = neuralnet.roll_weights(nn, gradients)
    j, _, _ = core.logistic_cost_function(X, activations[-1], Y)
    return j, dW_rolled, db_rolled

W, b = neuralnet.roll_weights(nn, weights)
costs, W, b = core.minimize_with_momentum(activation_cost_function, X, Y, W, b, learning_rate, 0.9, iterations)
# costs, W, b = core.minimize(activation_cost_function, X, Y, W, b, learning_rate, iterations)
weights = neuralnet.unroll_weights(nn, W, b)

results = nn.predict(X, weights)

""" Plot Results """
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3, 1)

""" Plot Results: Costs"""
axarr[0].plot(range(0, iterations), costs)
axarr[0].set_title('Gradient Descent')
axarr[0].set_xlabel('Iteration')
axarr[0].set_ylabel('Cost')

""" Plot Results: Original Data"""
Y_positive = numpy.array((0, n))
Y_negative = numpy.array((0, n))
for i in range(0, m):
    if(Y.T[i]) == 1:
        Y_positive = numpy.vstack((X.T[i, :2], Y_positive))
    else:
        Y_negative = numpy.vstack((X.T[i, :2], Y_negative))

axarr[1].plot(Y_positive[:, 0], Y_positive[:, 1], 'go')
axarr[1].plot(Y_negative[:, 0], Y_negative[:, 1], 'ro')
axarr[1].set_title('Data')
axarr[1].set_xlabel('x1')
axarr[1].set_ylabel('x2')

""" Plot Results: Predictions"""
R_positive = numpy.array((0, n))
R_negative = numpy.array((0, n))
for i in range(0, m):
    if(results[i]) == 1:
        R_positive = numpy.vstack((X.T[i, :2], R_positive))
    else:
        R_negative = numpy.vstack((X.T[i, :2], R_negative))

if len(numpy.shape(R_positive)) == 2:
    axarr[2].plot(R_positive[:, 0], R_positive[:, 1], 'go')

if len(numpy.shape(R_negative)) == 2:
    axarr[2].plot(R_negative[:, 0], R_negative[:, 1], 'ro')

axarr[2].set_title('Predictions')
axarr[2].set_xlabel('x1')
axarr[2].set_ylabel('x2')

f.subplots_adjust(hspace=0.5)
plt.show()