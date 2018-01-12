import numpy as np
from linear_svm import *
from softmax import *


class LinearClassifier:

    def __init__(self, batch_size=64, n_iters=1000, log_iters=100,
                 learning_rate=1e-3, initializer=None, regularization=5e-4):
        """
        :param batch_size: training batch_size, default 64
        :param n_iters: the number of iterations, default 1000
        :param log_iters: the number of logging iterations, default 100
        :param learning_rate: training learning rate, default 1e-3
        :param initializer: weight initializer, default None
        :param regularization: weight regularization, default 5e-4
        """

        self.batch_size = batch_size
        self.n_iters = n_iters
        self.log_iters = log_iters
        self.lr = learning_rate
        self.w_init = initializer
        self.reg = regularization

        self.x = None  # images
        self.y = None  # labels
        self.n_classes = 0  # the number of classes

        self.W = None  # Weights
        # self.b = None  # biases

    def train(self, x, y):
        dim, n_train = x.shape
        self.n_classes = np.max(y) + 1

        if not self.W:
            self.W = np.random.randn(self.n_classes, dim) * 0.001

        losses = []
        for i in range(self.n_iters):
            # Get training batch data
            rand_idx = np.random.choice(n_train, self.batch_size, replace=True)

            x_batch = x[:, rand_idx]
            y_batch = y[rand_idx]

            # Compute loss & gradient
            loss, gradient = self.loss(x_batch, y_batch, self.reg)
            losses.append(loss)

            # Update weights using gradients & learning rate
            self.W += - gradient * self.lr

            if i % self.log_iters == 0:
                print("[+] Iter %04d, loss : {.:8f}".format(i, np.mean(losses)))

        return losses

    def predict(self, x):
        scores = self.W.dot(x)
        y_prediction = scores.argmax(axis=0)

        return y_prediction

    def loss(self, x_batch, y_batch, reg):
        pass


class LinearSVM(LinearClassifier):

    def loss(self, x_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, x_batch, y_batch, reg)


class Softmax(LinearClassifier):

    def loss(self, x_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, x_batch, y_batch, reg)
