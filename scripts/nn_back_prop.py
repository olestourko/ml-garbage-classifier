from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from src.core import *
from src.neuralnet import *
import numpy
from numpy import shape, ones

# Load data
raw_data = numpy.loadtxt("../resources/ex2data1.txt", delimiter=',')
X = raw_data[:, 0:2]
Y = raw_data[:, -1:]
m = shape(X)[0] # number of training examples

# Generate some new features
x1_div_x2 = (X[:, 0] / X[:, 1]).reshape(m, 1)
X = numpy.concatenate((X, x1_div_x2), axis=1)

n = shape(X)[1] # number of features

bias_features = numpy.ones([m, 1])
X_with_bias = numpy.concatenate((bias_features, X), axis=1)
# theta_with_bias = randomly_intiate_theta_rolled(4, 1.0)
theta_with_bias = [
    ones([1, n+1]),
    # ones([1, 3]) # number of nodes, number of features. The bias node is added automatically after the first layer.
]

# Initialize a neural network
nn = NeuralNet(n)
nn.add_layer(n+1) # input layer
# nn.add_layer(3) # hidden layer
nn.add_layer(1) # output layer

# Randomly initialize thetas
random_theta = randomly_initiate_theta(theta_with_bias, 1)

# Unroll thetas
unrolled_theta = []
for key, value in enumerate(random_theta):
    unrolled_theta.extend(numpy.ravel(value).tolist())

# Train the neural network
alpha = 0.0001 # Gradient descent constant
rlambda = 0.0 # Regularization constant
n_iterations = 50000
trained_theta, costs = minimize(nn.backward_propagate, X, Y, unrolled_theta, alpha, rlambda, n_iterations)
results = nn.predict(X, trained_theta)

# Plot results
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3, 1)
# Cost over iterations
axarr[0].plot(range(0, n_iterations), costs)
axarr[0].set_title('Gradient Descent')
axarr[0].set_xlabel('Iteration')
axarr[0].set_ylabel('Cost')

# True values from data
Y_positive = numpy.array((0, n))
Y_negative = numpy.array((0, n))
for i in range(0, m):
    if(Y[i]) == 1:
        Y_positive = numpy.vstack((X[i, :2], Y_positive))
    else:
        Y_negative = numpy.vstack((X[i, :2], Y_negative))

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
        R_positive = numpy.vstack((X[i, :2], R_positive))
    else:
        R_negative = numpy.vstack((X[i, :2], R_negative))

if len(numpy.shape(R_positive)) == 2:
    axarr[2].plot(R_positive[:, 0], R_positive[:, 1], 'go')

if len(numpy.shape(R_negative)) == 2:
    axarr[2].plot(R_negative[:, 0], R_negative[:, 1], 'ro')
axarr[2].set_title('Predictions')
axarr[2].set_xlabel('x1')
axarr[2].set_ylabel('x2')

f.subplots_adjust(hspace=0.5)
plt.show()