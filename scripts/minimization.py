from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import src.core as core
import numpy
from numpy import shape, ones, zeros

# Load data
raw_data = numpy.loadtxt("../resources/ex2data1.txt", delimiter=',')
X = raw_data[:, 0:2].T
Y = raw_data[:, -1:].T
m = shape(X)[1] # number of training examples
n = shape(X)[0]

W, b = core.initiate_weights(n, 1, 1.0)

# Train weights
learning_rate = 0.0001
iterations = 5000
costs, W, b = core.minimize_2(X, Y, W, b, learning_rate, iterations)
results = core.predict_2(core.sigmoid, X, W, b).T

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
        Y_positive = numpy.vstack((X.T[i, :n], Y_positive))
    else:
        Y_negative = numpy.vstack((X.T[i, :n], Y_negative))

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
        R_positive = numpy.vstack((X.T[i, :n], R_positive))
    else:
        R_negative = numpy.vstack((X.T[i, :n], R_negative))

if len(numpy.shape(R_positive)) == 2:
    axarr[2].plot(R_positive[:, 0], R_positive[:, 1], 'go')

if len(numpy.shape(R_negative)) == 2:
    axarr[2].plot(R_negative[:, 0], R_negative[:, 1], 'ro')

axarr[2].set_title('Predictions')
axarr[2].set_xlabel('x1')
axarr[2].set_ylabel('x2')

f.subplots_adjust(hspace=0.5)
plt.show()