from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import src.core as core
import numpy
from numpy import shape, ones, zeros
import data_loader

X, Y = data_loader.get_sample_data()
m = shape(X)[1]
n = shape(X)[0]

W, b = core.initialize_weights(n, 1, 1.0)

# Train weights
learning_rate = 0.008
iterations = 5000
# costs, W, b = core.minimize_2(X, Y, W, b, learning_rate, iterations)

def activation_cost_function(X, Y, W, b):
    z = core.calculate_z(X, W, b)
    a = core.sigmoid(z)
    j, dW, db = core.logistic_cost_function(X, a, Y)
    return j, dW.T, db.T # Transposing the gradients because I changed the minimization functions to work with NNs

costs, W, b = core.minimize_with_momentum(
    activation_cost_function,
    X, Y, W, b, learning_rate, 0.9, iterations
)

results = core.predict(core.sigmoid, X, W, b).T

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