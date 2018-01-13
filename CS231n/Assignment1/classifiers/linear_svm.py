import numpy as np


def svm_loss_naive(w, x, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on mini-batches of N examples.

    :param w: A numpy array of shape (D, C) containing weights.
    :param x: A numpy array of shape (N, D) containing a mini-batch of data.
    :param y: A numpy array of shape (N,) containing training labels;
    y[i] = c means that X[i] has label c, where 0 <= c < C.
    :param reg: (float) regularization strength
    :return: loss (single float), gradient (an array of same shape as W)
    """

    dw = np.zeros(w.shape)  # initialize the gradient as zero

    n_classes = w.shape[1]
    n_train = x.shape[0]

    loss = 0.

    # compute loss & gradient
    for i in range(n_train):
        scores = x[i].dot(w)
        correct_scores = scores[y[i]]

        for j in range(n_classes):
            if j == y[i]:
                continue

            margin = scores[j] - correct_scores + 1.

            if margin > 0.:
                loss += margin

                # update gradient
                dw[:, j] += x[i]
                dw[:, y[i]] -= x[i]

    loss /= n_train
    dw /= n_train

    # L2 regularization
    w_reg = .5 * reg * float(np.tensordot(w, w, axes=((0, 1), (0, 1))))
    dw_reg = reg * w  # differential of w_reg by w

    # add reg to the loss and gradient
    loss += w_reg
    dw += dw_reg

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dw


def svm_loss_vectorized(w, x, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.

    :param w: Same as above
    :param x: Same as above
    :param y: Same as above
    :param reg: Same as above
    :return: Same as above
    """

    n_train = x.shape[0]
    delta = 1

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    scores = x.dot(w)
    idx_correct_scores = [np.arange(n_train), y]
    correct_scores = scores[idx_correct_scores[0], idx_correct_scores[1]].reshape(-1, 1)

    margins = np.maximum(0, scores - correct_scores + delta)
    margins[idx_correct_scores[0], idx_correct_scores[1]] = 0

    # L2 regularization
    w_reg = .5 * reg * float(np.tensordot(w, w, axes=((0, 1), (0, 1))))

    # SVM loss with regularization
    loss = np.sum(margins) / float(n_train) + w_reg

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    margins[np.where(margins > 0)] = 1
    margins[range(n_train), y] = -np.sum(margins, axis=1)

    # L2 regularization
    dw_reg = reg * w

    dw = x.T.dot(margins)
    dw = dw / float(n_train) + dw_reg

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dw
