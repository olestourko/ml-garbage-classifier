from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import numpy

def get_sample_data(normalized=False):
    raw_data = numpy.loadtxt(path.dirname(path.dirname(path.abspath(__file__))) + "/resources/ex2data1.txt", delimiter=',')
    X = raw_data[:, 0:2].T
    # Add an extra feature
    ratio_feature = X[0, :] / X[1, :]
    X = numpy.vstack((X, ratio_feature))

    Y = raw_data[:, -1:].T
    m = numpy.shape(X)[1] # number of training examples
    n = numpy.shape(X)[0]

    if normalized:
        mean = X.mean(axis=1)
        variance = (1.0 / m) * numpy.sum(X ** 2, axis=1)
        for column in range(0, X.shape[0]):
            X[column, :] -= mean[column]
            X[column, :] /= variance[column]

    return X, Y