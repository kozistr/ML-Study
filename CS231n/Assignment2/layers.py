import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return out, cache:
        - out: output, of shape (N, M),
        - cache: (x, w, b)
    """

    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    x = np.reshape(x, (x.shape[0], -1))  # flatten, (N, D)

    out = x.dot(w) + np.column_stack(b)  # (N, M) + (, M)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)

    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Input data, of shape (N, d_1, ... d_k), Weights, of shape (D, M)

    :return dx, dw, db:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k),
        - dw: Gradient with respect to w, of shape (D, M),
        - db: Gradient with respect to b, of shape (M,)
    """

    x, w, b = cache
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################

    x_dim = x.shape[1:]
    n, m = dout.shape
    d = np.prod(x_dim)

    x = np.reshape(x, (n, d))  # (N, D)

    dx = dout.dot(w.T)  # (N, M) * (M, D) = (N, D)
    dw = x.T.dot(dout)  # (D, N) * (N, M) = (D, M)
    db = np.sum(dout)

    dx = dx.reshape((n, ) + x_dim)  # (N, d1, ..., d_k)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    :param x: Inputs, of any shape

    :return out, cache:
        - out: Output, of the same shape as x,
        - cache: x
    """

    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################

    out = np.maximum(0., x)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    :param dout: Upstream derivatives, of any shape
    :param cache: Input x, of same shape as dout

    :return dx: Gradient with respect to x
    """

    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    x[x <= 0.] = 0.
    x[x > 0.] = 1

    dx = x * dout

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from mini-batch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each time-step we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    :param x: Data of shape (N, D)
    :param gamma: Scale parameter of shape (D,)
    :param beta: Shift parameter of shape (D,)
    :param bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features

    :return out, cache: f shape (N, D), A tuple of values needed in the backward pass
    """

    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use mini-batch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################

        sample_mean = np.mean(x)
        sample_var = np.var(x)

        # normalize
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)

        out = gamma * x_norm + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache['x'] = x
        cache['gamma'] = gamma
        cache['beta'] = beta
        cache['eps'] = eps
        cache['sample_mean'] = sample_mean
        cache['sample_var'] = sample_var
        cache['x_normalized'] = x_norm

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        # sample_mean = np.mean(x)
        # sample_var = np.var(x)

        # normalize
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)

        out = gamma * x_norm + beta

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    :param dout: Upstream derivatives, of shape (N, D)
    :param cache: Variable of intermediates from batchnorm_forward.

    :return dx, dgamma, dbeta:
        - dx: Gradient with respect to inputs x, of shape (N, D),
        - dgamma: Gradient with respect to scale parameter gamma, of shape (D,),
        - dbeta: Gradient with respect to shift parameter beta, of shape (D,),
    """

    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################

    n, d = dout.shape

    x = cache['x']
    x_norm = cache['x_normalized']
    gamma = cache['gamma']
    # beta = cache['beta']
    eps = cache['eps']

    sample_mean = cache['sample_mean']
    sample_var = cache['sample_var']

    d_x_norm = dout * gamma

    d_sample_var = np.sum(d_x_norm * (x - sample_mean) * (-.5) * (sample_var + eps) ** (-1.5))
    d_sample_mean = np.sum(d_x_norm * (-1) * (sample_var + eps) ** (-.5)) + \
        d_sample_var * (-2. / n) * np.sum(x - sample_mean)

    dx = d_x_norm * (sample_var + eps) ** (-.5) + d_sample_mean * (1. / n) + d_sample_var * (2. / n) * (x - sample_mean)
    dgamma = np.sum(x_norm * dout)
    dbeta = np.sum(dout)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    :param dout: Same as batchnorm_backward
    :return cache: Same as batchnorm_backward
    """

    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    n, d = dout.shape

    x = cache['x']
    x_norm = cache['x_normalized']
    gamma = cache['gamma']
    # beta = cache['beta']
    eps = cache['eps']

    sample_mean = cache['sample_mean']
    sample_var = cache['sample_var']

    dx = (gamma / np.sqrt(sample_var + eps)) * \
         (dout - np.mean(dout) - ((x - sample_mean) / (n * (sample_var + eps))) * np.sum(dout * (x - sample_mean)))
    dgamma = np.sum(dout * x_norm)
    dbeta = np.sum(dout)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    :param x: Input data, of any shape
    :param dropout_param: A dictionary with the following keys:
        - p: Dropout parameter. We drop each neuron output with probability p.
        - mode: 'test' or 'train'. If the mode is train, then perform dropout;
          if the mode is test, then just return the input.
        - seed: Seed for the random number generator. Passing seed makes this
          function deterministic, which is needed for gradient checking but not
          in real networks.

    :return out, cache:
        - out : Array of the same shape as x,
        - cache : tuple (dropout_param, mask).
            In training mode, mask is the dropout mask that was used to multiply the input;
            in test mode, mask is None.
    """

    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################

        # mask = (np.random.rand(*x.shape) < p)
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask  # drop

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################

        # out = x * p
        out = x

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    :param dout: Upstream derivatives, of any shape
    :param cache: (dropout_param, mask) from dropout_forward.
    """

    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################

        dx = mask * dout

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout

    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    :param x: Input data of shape (N, C, H, W)
    :param w: Filter weights of shape (F, C, HH, WW)
    :param b: Biases, of shape (F,)
    :param conv_param: A dictionary with the following keys:
        - 'stride': The number of pixels between adjacent receptive fields in the
          horizontal and vertical directions.
        - 'pad': The number of pixels that will be used to zero-pad the input.

    :return out, cache: Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
    """

    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    n, c, h, w = x.shape
    f, c, hh, ww = w.shape

    stride = conv_param['stride']
    pad = conv_param['pad']

    # zero padding
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant', constant_values=(0, 0, 0, 0))

    h_ = 1 + (h + 2 * pad - hh) / stride
    w_ = 1 + (h + 2 * pad - ww) / stride

    out = np.zeros((n, f, h_, w_))

    for i in range(n):
        for j in range(f):
            for k in range(h_):
                for l in range(w_):
                    out[i, j, k, l] = np.sum(
                        x_pad[n, :, k * stride:k * stride + hh, l * stride:l * stride + ww] * w[f, :]
                    ) + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)

    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    :param dout: Upstream derivatives.
    :param cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    :return dx, dw, db: Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
    """

    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    :param x: Input data, of shape (N, C, H, W)
    :param pool_param: dictionary with the following keys:
        - 'pool_height': The height of each pooling region
        - 'pool_width': The width of each pooling region
        - 'stride': The distance between adjacent pooling regions

    :return out, cahce: Returns a tuple of:
        - out: Output data
        - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)

    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    :param dout: Upstream derivatives
    :param cache: A tuple of (x, pool_param) as in the forward pass.

    :return dx: Gradient with respect to x
    """

    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    :param x: Input data of shape (N, C, H, W)
    :param gamma: Scale parameter, of shape (C,)
    :param beta: Shift parameter, of shape (C,)
    :param bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance. momentum=0 means that
          old information is discarded completely at every time step, while
          momentum=1 means that new information is never incorporated. The
          default of momentum=0.9 should work well in most situations.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features

    :return out, cache: Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
    """

    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    :param dout: Upstream derivatives, of shape (N, C, H, W)
    :param cache: Values from the forward pass

    :return dx, dgamma, dbeta: Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """

    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    :param x: Input data, of shape (N, C) where x[i, j] is the score for the jth class for the ith input.
    :param y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and 0 <= y[i] < C

    :return loss, dx: Returns a tuple of:
        - loss: Scalar giving the loss
        - dx: Gradient of the loss with respect to x
    """

    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N

    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    :param x: Input data, of shape (N, C) where x[i, j] is the score for the jth class for the ith input.
    :param y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and 0 <= y[i] < C

    :return loss, dx: Returns a tuple of:
        - loss: Scalar giving the loss
        - dx: Gradient of the loss with respect to x
    """

    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx
