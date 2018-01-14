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

    n_train = x.shape[0]

    loss = 0.

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    for i in range(n_train):
        score_i = x[i].dot(w)
        stability = -np.max(score_i)

        scores = np.exp(score_i + stability)
        scores_norm = scores / float(np.sum(scores))

        loss += -np.log(scores_norm[y[i]])

        """
        for j in range(n_classes):
            dw[:. j] += (score_i[j] / np.sum(scores)) * x.T

            if j == y[i]:
                dw[:. j] -= x.T  # remove j == y[i] case
        """

        dw_update = np.outer(x[i], scores_norm)
        dw_update[:, y[i]] -= x[i]

        dw += dw_update

    # L2 regularization
    w_reg = .5 * reg * float(np.tensordot(w, w, axes=((0, 1), (0, 1))))
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

    n_train = x.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    scores = x.dot(w)
    instability = -np.row_stack(np.max(scores, axis=1))

    scores += instability  # solve numeric instability
    scores = np.exp(scores)
    scores /= np.row_stack(np.sum(scores, axis=1)).astype(float)

    # L2 regularization
    w_reg = .5 * reg * float(np.tensordot(w, w, axes=((0, 1), (0, 1))))
    dw_reg = reg * w  # differential of w_reg by w

    loss = np.sum(-np.log(scores[np.arange(n_train), y])) / float(n_train) + w_reg

    dw = scores
    dw[np.arange(n_train), y] -= 1
    dw = x.T.dot(dw) / float(n_train) + dw_reg

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dw
