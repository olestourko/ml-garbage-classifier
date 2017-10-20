import numpy
import src.core as core
import src.utils as utils

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
    weight_vector = utils.weights_to_vector(W, b)
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
        _W, _b = utils.vector_to_weights(thetaplus, input_size, output_size)
        a = activate(X, _W, _b)
        jplus, _, _ = core.logistic_cost_function(X, a, Y)
        _W, _b = utils.vector_to_weights(thetaminus, input_size, output_size)
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
    dg_vector = utils.weights_to_vector(dW, db)

    diff = (
        numpy.linalg.norm(dg_vector - ncg_vector) /
        (numpy.linalg.norm(dg_vector) + numpy.linalg.norm(ncg_vector))
    )
    return diff

def gradient_check_nn(nn, X, weights, Y, epsilon=1e-7):
    weight_vector = utils.nn_weights_to_vector(weights)

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
        _weights = utils.vector_to_nn_weights(thetaplus, nn.layers)
        activations, _ = nn.forward_propagate(X, _weights)
        jplus, _, _ = core.logistic_cost_function(X, activations[-1], Y)
        _weights = utils.vector_to_nn_weights(thetaminus, nn.layers)
        activations, _ = nn.forward_propagate(X, _weights)
        jminus, _, _ = core.logistic_cost_function(X, activations[-1], Y)
        ncg_vector[i] = (jplus - jminus) / (2.0 * epsilon)

    """
    Compute the gradients using differentiation
    """
    activations, zs = nn.forward_propagate(X, weights)
    # Differentiated gradients
    gradients = nn.backward_propagate(X, activations, zs, Y, weights)
    dg_vector = utils.nn_weights_to_vector(gradients)

    diff = (
        numpy.linalg.norm(dg_vector - ncg_vector) /
        (numpy.linalg.norm(dg_vector) + numpy.linalg.norm(ncg_vector))
    )
    return diff
