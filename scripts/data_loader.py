import numpy

def get_sample_data():
    raw_data = numpy.loadtxt("../resources/ex2data1.txt", delimiter=',')
    X = raw_data[:, 0:2].T
    # Add an extra feature
    ratio_feature = X[0, :] / X[1, :]
    X = numpy.vstack((X, ratio_feature))

    Y = raw_data[:, -1:].T
    m = numpy.shape(X)[1] # number of training examples
    n = numpy.shape(X)[0]

    return X, Y