import numpy as np


def svm_loss_naive(w, x, y, reg):
    """
    Structured Multi-Class SVM loss function, naive implementation
    :param w: weights
    :param x: x images
    :param y: y labels
    :param reg: weights regularization
    :return: loss, gradient
    """

    dw = np.zeros(w.shape)

    n_classes = w.shape[0]
    n_train = x.shape[1]

    loss = 0.

    # compute loss & gradient
    for i in range(n_train):
        scores = w.dot(x[:, i])
        correct_scores = scores[y[i]]

        indicator = (scores - correct_scores + 1) > 0.

        for j in range(n_classes):
            if j == y[i]:
                dw[j, :] = -np.sum(np.delete(indicator, j)) * x[:, i].T
                continue

            dw[j, :] += indicator[j] * x[:, i].T
            margin = scores[j] - correct_scores + 1.

            if margin > 0.:
                loss += margin

    loss /= n_train
    dw /= n_train

    # L2 regularization
    regularization = .5 * reg * np.sum(np.power(w, 2))

    # add reg to the loss and gradient
    loss += regularization
    dw += reg * w

    return loss, dw


def svm_loss_vectorized(w, x, y, reg):
    """
    Structured Multi-Class SVM loss function, vectorized implementation
    :param w: weights
    :param x: x images
    :param y: y labels
    :param reg: weights regularization
    :return: loss, gradient
    """

    n_train = x.shape[1]

    loss = w.dot(x) - (w.dot(x))[y, np.arange(n_train)] + 1.
    indicator = loss > 0.

    loss = np.sum(loss * indicator, axis=0) - 1.

    # L2 regularization
    regularization = .5 * reg * np.sum(np.power(w, 2))
    loss = np.sum(loss) / float(n_train) + regularization

    indicator *= np.ones(loss.shape)
    indicator[[y, np.arange(n_train)]] = -(np.sum(indicator, axis=0) - 1.)

    dw = indicator.dot(x.T) / float(n_train)
    dw += reg * w

    return loss, dw
