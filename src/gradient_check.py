import numpy
import core

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

def gradient_check_simple_logistic(X, Y, W, b, epsilon=1e-7):
    """

    :param X:
    :param Y:
    :param W:
    :param b:
    :param epsilon: The bump value used when numerical
     computing each derivative
    :return: dW, db
    """

    input_size = W.shape[0]
    output_size = W.shape[1]
    weight_vector = weights_to_vector(W, b)
    """
    Compute the gradients numerically
    
    https://www.coursera.org/learn/deep-neural-network/lecture/XzSSa/numerical-approximation-of-gradients
    http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
    
    Also worth checking out the steps for gradient checking in the Jupyter notebook
    """
    def activate(X, W, b):
        z = core.calculate_z(X, W, b)
        a = core.sigmoid(z)
        return a

    # Numerically computed gradients
    ncg_vector = numpy.zeros(weight_vector.shape)

    for i in range(0, weight_vector.shape[0]):
        e = numpy.zeros(weight_vector.shape[0])
        e[i] = epsilon
        thetaplus = weight_vector + e
        thetaminus = weight_vector - e
        _W, _b = vector_to_weights(thetaplus, input_size, output_size)
        a = activate(X, _W, _b)
        jplus, _, _ = core.logistic_cost_function(X, a, Y)
        _W, _b = vector_to_weights(thetaminus, input_size, output_size)
        a = activate(X, _W, _b)
        jminus, _, _ = core.logistic_cost_function(X, a, Y)
        ncg_vector[i] = (jplus - jminus) / (2.0 * epsilon)

    """
    Compute the gradients using differentiation
    """
    def activation_cost_function(X, Y, W, b):
        z = core.calculate_z(X, W, b)
        a = core.sigmoid(z)
        j, dW, db = core.logistic_cost_function(X, a, Y)
        return j, dW.T, db.T  # Transposing the gradients because I changed the minimization functions to work with NNs

    # Differentiated gradients
    j, dW, db = activation_cost_function(X, Y, W, b)
    dg_vector = weights_to_vector(dW, db)

    diff = (
        numpy.linalg.norm(dg_vector - ncg_vector) /
        (numpy.linalg.norm(dg_vector) + numpy.linalg.norm(ncg_vector))
    )
    return diff

def gradient_check_nn(nn, X, weights, Y, epsilon=1e-7):
    weight_vector = nn_weights_to_vector(weights)

    # Numerically computed gradients
    ncg_vector = numpy.zeros(weight_vector.shape)

    """
    Compute the gradients numerically
    """
    for i in range(0, weight_vector.shape[0]):
        e = numpy.zeros(weight_vector.shape[0])
        e[i] = epsilon
        thetaplus = weight_vector + e
        thetaminus = weight_vector - e
        _weights = vector_to_nn_weights(thetaplus, nn.layers)
        activations, _ = nn.forward_propagate(X, _weights)
        jplus, _, _ = core.logistic_cost_function(X, activations[-1], Y)
        _weights = vector_to_nn_weights(thetaminus, nn.layers)
        activations, _ = nn.forward_propagate(X, _weights)
        jminus, _, _ = core.logistic_cost_function(X, activations[-1], Y)
        ncg_vector[i] = (jplus - jminus) / (2.0 * epsilon)

    """
    Compute the gradients using differentiation
    """
    activations, zs = nn.forward_propagate(X, weights)
    # Differentiated gradients
    gradients = nn.backward_propagate(X, activations, zs, Y, weights)
    dg_vector = nn_weights_to_vector(gradients)

    diff = (
        numpy.linalg.norm(dg_vector - ncg_vector) /
        (numpy.linalg.norm(dg_vector) + numpy.linalg.norm(ncg_vector))
    )
    return diff
