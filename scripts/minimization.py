from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import src.core as core
import numpy
from numpy import shape, ones, zeros

# Load data
raw_data = numpy.loadtxt("../resources/ex2data1.txt", delimiter=',')
X = raw_data[:, 0:2]
Y = raw_data[:, -1:]
m = shape(X)[0] # number of training examples
n = shape(X)[1] # number of features

bias_features = numpy.ones([m, 1])
X_with_bias = numpy.concatenate((bias_features, X), axis=1)
theta_with_bias = zeros([1, n+1])
# theta_with_bias = numpy.array([
#     [-24, 0.2, 0.2]
# ])

# Test cost function
cost, gradients = core.logisticCostFunction(X_with_bias, Y, 0, theta_with_bias.transpose())

# Train thetas
n_iterations = 400
alpha = 0.0001
rlambda = 0
trained_theta, costs = core.minimize(core.logisticCostFunction, X_with_bias, Y, theta_with_bias.transpose(), alpha, rlambda, n_iterations)
results = core.predict(X_with_bias, trained_theta)

# Plot results
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3, 1)
axarr[0].plot(range(0, n_iterations), costs)
axarr[0].set_title('Gradient Descent')
axarr[0].set_xlabel('Iteration')
axarr[0].set_ylabel('Cost')

# True values from data
Y_positive = numpy.array((0, n))
Y_negative = numpy.array((0, n))
for i in range(0, m):
    if(Y[i]) == 1:
        Y_positive = numpy.vstack((X[i], Y_positive))
    else:
        Y_negative = numpy.vstack((X[i], Y_negative))

axarr[1].plot(Y_positive[:, 0], Y_positive[:, 1], 'go')
axarr[1].plot(Y_negative[:, 0], Y_negative[:, 1], 'ro')
axarr[1].set_title('Data')
axarr[1].set_xlabel('x1')
axarr[1].set_ylabel('x2')

# Results
R_positive = numpy.array((0, n))
R_negative = numpy.array((0, n))
for i in range(0, m):
    if(results[i]) == 1:
        R_positive = numpy.vstack((X[i], R_positive))
    else:
        R_negative = numpy.vstack((X[i], R_negative))

if len(numpy.shape(R_positive)) == 2:
    axarr[2].plot(R_positive[:, 0], R_positive[:, 1], 'go')

if len(numpy.shape(R_negative)) == 2:
    axarr[2].plot(R_negative[:, 0], R_negative[:, 1], 'ro')

axarr[2].set_title('Predictions')
axarr[2].set_xlabel('x1')
axarr[2].set_ylabel('x2')

f.subplots_adjust(hspace=0.5)
plt.show()
plt.show()