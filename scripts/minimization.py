from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import src.core as core
import numpy
from numpy import shape, ones

# Load data
raw_data = numpy.loadtxt("../resources/ex2data2.txt", delimiter=',')
X = raw_data[:, 0:2]
Y = raw_data[:, -1:]
m = shape(X)[0] # number of training examples
n = shape(X)[1] # number of features
theta = ones([1, n])

bias_features = numpy.ones([m, 1])
X_with_bias = numpy.concatenate((bias_features, X), axis=1)
theta_with_bias = ones([1, n+1])

# Train thetas
n_iterations = 100
trained_theta, costs = core.minimize(core.logisticCostFunction, X_with_bias, Y, theta_with_bias.transpose(), 0.5, n_iterations)

# Plot results
import matplotlib.pyplot as plt
f, axarr = plt.subplots(2, 1)
axarr[0].plot(range(0, n_iterations), costs)
axarr[0].set_title('Gradient Descent')
axarr[0].set_xlabel('Iteration')
axarr[0].set_ylabel('Cost')

X_positive = numpy.array((0, n))
X_negative = numpy.array((0, n))
for i in range(0, m):
    if(Y[i]) == 1:
        X_positive = numpy.vstack((X[i], X_positive))
    else:
        X_negative = numpy.vstack((X[i], X_negative))

axarr[1].plot(X_positive[:, 0], X_positive[:, 1], 'go')
axarr[1].plot(X_negative[:, 0], X_negative[:, 1], 'ro')
axarr[1].set_title('Data')
axarr[1].set_xlabel('x1')
axarr[1].set_ylabel('x2')
f.subplots_adjust(hspace=0.5)
plt.show()