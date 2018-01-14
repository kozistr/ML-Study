import numpy as np
from classifiers.linear_svm import *
from classifiers.softmax import *


class LinearClassifier(object):

    def __init__(self, batch_size=200, n_iters=1000, log_iters=100,
                 learning_rate=1e-3, regularization=5e-4):
        """
        Train this linear classifier using stochastic gradient descent.

        :param batch_size: (integer) number of training examples to use at each step, default 200
        :param n_iters: (integer) number of steps to take when optimizing, default 1000
        :param log_iters: (integer) number of steps to log when optimizing, default 100
        :param learning_rate: (float) learning rate for optimization., default 1e-3
        :param regularization: (float) regularization strength., default 5e-4
        """

        self.batch_size = batch_size
        self.n_iters = n_iters
        self.log_iters = log_iters
        self.lr = learning_rate
        self.reg = regularization

        self.x = None  # images
        self.y = None  # labels
        self.n_classes = 0  # the number of classes

        self.W = None  # Weights

    def train(self, x, y):
        """
        :param x: A numpy array of shape (N, D) containing training data;
        there are N training samples each of dimension D.
        :param y: A numpy array of shape (N,) containing training labels;
        y[i] = c means that X[i] has label 0 <= c < C for C classes.
        :return: A list containing the value of the loss function at each training iteration.
        """

        n_train, dim = x.shape
        self.n_classes = int(np.max(y) + 1)  # assume y takes values 0...K-1 where K is number of classes

        if not self.W:
            self.W = np.random.randn(dim, self.n_classes) * 0.001  # lazily initialize W

        # Run stochastic gradient descent to optimize W
        losses = []
        for i in range(self.n_iters):
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################

            # Get training batch data
            rand_idx = np.random.choice(np.arange(n_train), self.batch_size, replace=True)

            x_batch = x[rand_idx]
            y_batch = y[rand_idx]

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # Compute loss & gradient
            loss, gradient = self.loss(x_batch, y_batch, self.reg)
            losses.append(loss)

            # Update weights using gradients & learning rate
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################

            self.W += - gradient * self.lr

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if i % self.log_iters == 0:
                print("[+] Iter {}, loss : {}".format(i, loss))

        return losses

    def predict(self, x):
        """
        Use the trained weights of this linear classifier to predict labels for data points.

        :param x: A numpy array of shape (N, D) containing training data;
        there are N training samples each of dimension D.
        :return: Predicted labels for the data in X. y_pred is a 1-dimensional array of length N,
        and each element is an integer giving the predicted class.
        """

        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################

        scores = np.dot(x, self.W)
        y_prediction = np.argmax(scores, axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################

        return y_prediction

    def loss(self, x_batch, y_batch, reg):
        """
        Compute the loss function and its derivative. Subclasses will override this.

        :param x_batch: A numpy array of shape (N, D) containing a minibatch of N data points;
        each point has dimension D.
        :param y_batch: A numpy array of shape (N,) containing labels for the mini-batch.
        :param reg: (float) regularization strength.
        :return: loss (single float), gradient (an array of same shape as W)
        """

        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, x_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, x_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, x_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, x_batch, y_batch, reg)
