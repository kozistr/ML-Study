import numpy as np


def softmax_loss_naive(w, x, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on mini-batches of N examples.

    :param w: A numpy array of shape (D, C) containing weights.
    :param x: A numpy array of shape (N, D) containing a mini-batch of data.
    :param y: A numpy array of shape (N,) containing training labels;
    y[i] = c means that X[i] has label c, where 0 <= c < C.
    :param reg: (float) regularization strength
    :return: loss (single float), gradient (an array of same shape as W)
    """

    dw = np.zeros(w.shape)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    n_train = x.shape[1]
    n_classes = dw.shape[0]

    loss = 0.

    for i in range(n_train):
        x_i = x[:, i]
        score_i = w.dot(x_i)

        stability = -np.max(score_i)

        exp_score_i = np.exp(score_i + stability)
        exp_score_sum_i = np.sum(exp_score_i, axis=0)

        for j in range(n_classes):
            dw[j, :] += (score_i[j] / exp_score_sum_i) * x.T

            if j == y[i]:
                dw[j, :] -= x.T  # remove j == y[i] case

        numerator = np.exp(score_i[y[i]] + stability)

        loss += -np.log(numerator / float(exp_score_sum_i))

    # L2 regularization
    w_reg = .5 * reg * np.sum(np.power(w, 2))
    dw_reg = reg * w  # differential of w_reg by w

    loss = loss / float(n_train) + w_reg
    dw = dw / float(n_train) + dw_reg

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dw


def softmax_loss_vectorized(w, x, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.

    :param w: Same as above
    :param x: Same as above
    :param y: Same as above
    :param reg: Same as above
    :return: Same as above
    """

    # dw = np.zeros(w.shape)

    n_train = x.shape[1]
    # n_classes = dw.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    score = w.dot(x)
    stability = -np.max(score, axis=0)

    score = score + stability
    exp_score = np.exp(score)
    exp_score_sum = np.sum(exp_score, axis=0)

    # L2 regularization
    w_reg = .5 * reg * np.sum(np.power(w, 2))
    dw_reg = reg * w  # differential of w_reg by w

    loss = np.log(exp_score_sum) - score[y, np.arange(n_train)]
    loss = np.sum(loss) / float(n_train) + w_reg

    dw = exp_score / exp_score_sum
    dw[y, np.arange(n_train)] += -1.
    dw = dw.dot(x.T) / float(n_train) + dw_reg

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dw
