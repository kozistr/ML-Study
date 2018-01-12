import numpy as np


def softmax_loss_naive(w, x, y, reg):
    """
    softmax loss function, naive implementation
    :param w: weights
    :param x: x images
    :param y: y labels
    :param reg: weights regularization
    :return: loss, gradient
    """

    dw = np.zeros(w.shape)

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
    regularization = .5 * reg * np.sum(np.power(w, 2))

    loss = loss / float(n_train) + regularization
    dw = dw / float(n_train) + reg * w

    return loss, dw


def softmax_loss_vectorized(w, x, y, reg):
    """
    softmax loss function, vectorized implementation
    :param w: weights
    :param x: x images
    :param y: y labels
    :param reg: weights regularization
    :return: loss, gradient
    """

    # dw = np.zeros(w.shape)

    n_train = x.shape[1]
    # n_classes = dw.shape[0]

    score = w.dot(x)
    stability = -np.max(score, axis=0)

    score = score + stability
    exp_score = np.exp(score)
    exp_score_sum = np.sum(exp_score, axis=0)

    # L2 regularization
    regularization = .5 * reg * np.sum(np.power(w, 2))

    loss = np.log(exp_score_sum) - score[y, np.arange(n_train)]
    loss = np.sum(loss) / float(n_train) + regularization

    dw = exp_score / exp_score_sum
    dw[y, np.arange(n_train)] += -1.
    dw = dw.dot(x.T) / float(n_train) + reg * w

    return loss, dw
