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

    dw = np.zeros_like(w)

    n_train = x.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    score = x.dot(w)
    stability = -np.max(score)

    exp_score = np.exp(score + stability)
    exp_score_sum = np.sum(exp_score)

    # L2 regularization
    w_reg = .5 * reg * float(np.tensordot(w, w, axes=((0, 1), (0, 1))))
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
