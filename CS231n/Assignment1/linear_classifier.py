import numpy as np


class LinearClassifier:

    def __init__(self, batch_size=64, learning_rate=1e-3, initializer=None, regularization=5e-4):
        """
        :param batch_size: training batch_size, default 64
        :param learning_rate: training learning rate, default 1e-3
        :param initializer: weight initializer, default None
        :param regularization: weight regularization, default 5e-4
        """

        self.batch_size = batch_size
        self.lr = learning_rate
        self.w_init = initializer
        self.reg = regularization

        self.x = None  # images
        self.y = None  # labels
        self.n_classes = 0  # the number of classes

        self.W = None  # Weights
        # self.b = None  # biases

    def train(self, x, y):
        self.n_classes = np.max(y) + 1

        if not self.W:
            self.W = 0.
