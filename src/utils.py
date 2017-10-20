import numpy

def weights_to_vector(W, b):
    """
    Flattens simple logistic regression weights to a vector

    :param W: A numpy matrix: n-1 x n (layer input size x layer output size)
    :param b: A numpy matrix: n x 1 (layer output size)
    :return: a flat list of weights [w1, w2... b1, b2..]
    """

    thetas = W.flatten()
    thetas = numpy.append(thetas, b.flatten())
    return numpy.array(thetas)


def vector_to_weights(v, input_size, output_size):
    """
    Expands a weight vector

    :param v:
    :param input_size:
    :param output_size:
    :return: W, b tuple
    """

    W = numpy.array(v[0:input_size * output_size]).reshape(input_size, output_size)
    b = numpy.array(v[input_size * output_size:(input_size * output_size) + output_size]).reshape(output_size, 1)
    return W, b


def nn_weights_to_vector(weights):
    """
    Flattens neural network weights to a vector

    :param weights: A list of W, b dicts (one tuple for each layer)
    :return: The flattened weight vector
    """

    thetas = []
    for weight_tuple in weights:
        flattened_weight_tuple = weights_to_vector(weight_tuple["W"], weight_tuple["b"])
        thetas = numpy.append(thetas, flattened_weight_tuple)

    return numpy.array(thetas)

def vector_to_nn_weights(v, layers):
    """
    Expands a weight vector to a weight dict for a neural network

    :param v: The flattened weight vector
    :param layers: A list of layer sizes (including the input layer)
    :return: A list of W, b dicts (one tuple for each layer)
    """
    weights = []
    slice_from = 0
    for i in range(1, len(layers)):
        input_size = layers[i - 1]
        output_size = layers[i]
        slice_to = slice_from + (input_size * output_size) + output_size
        vector_section = v[slice_from:slice_to]
        W, b = vector_to_weights(vector_section, input_size, output_size)
        weights.append({
            "W": W,
            "b": b
        })
        slice_from = slice_to

    return weights