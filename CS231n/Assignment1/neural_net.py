import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of N,
    a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU non-linearity after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        :param input_size: The dimension D of the input data.
        :param hidden_size: The number of neurons H in the hidden layer.
        :param output_size: The number of classes C.
        :param std: standard error regulator, default 1e-4
        """

        self.params = dict()

        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, x, y=None, reg=0.):
        """
        Compute the loss and gradients for a two layer fully connected neural network.

        :param x: Input data of shape (N, D). Each X[i] is a training sample.
        :param y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
        an integer in the range 0 <= y[i] < C. This parameter is optional; if it
        is not passed then we only return scores, and if it is passed then we
        instead return the loss and gradients.
        :param reg: (float) Regularization strength.
        :return:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters with respect to the loss function;
         has the same keys as self.params.
        """

        # Unpack variables from the params dictionary
        w1, b1 = self.params['W1'], self.params['b1']
        w2, b2 = self.params['W2'], self.params['b2']
        n, d = x.shape

        # Compute the forward pass
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################

        h1 = x.dot(w1) + b1       # fc layer
        a1 = np.maximum(0., h1)    # ReLU Activation

        scores = a1.dot(w2) + b2  # fc layer

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss. So that your results match ours, multiply the            #
        # regularization loss by 0.5                                                #
        #############################################################################

        exp_scores = np.exp(scores)
        prob = exp_scores / np.sum(scores, axis=1, keepdims=True)  # softmax loss

        # Average cross-entropy loss & regularization
        correct_log_prob = -np.log(prob[y, np.arange(n)])
        data_loss = np.sum(correct_log_prob) / n

        w1_reg = .5 * reg * np.sum(np.power(w1, 2))
        w2_reg = .5 * reg * np.sum(np.power(w2, 2))
        w_reg = w1_reg + w2_reg

        loss = data_loss + w_reg

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################

        # compute the gradient on scores
        d_scores = prob
        d_scores[y, np.arange(n)] -= 1.
        d_scores /= n

        # back-prop w1, b1
        grads['W2'] = np.dot(a1.T, d_scores)
        grads['b2'] = np.sum(d_scores, axis=0)

        # back-prop hidden layer
        d_hidden = np.dot(d_scores, w2.T)

        # back-prop ReLU Activation
        d_hidden[a1 <= 0.] = 0.

        # back-prop w1, b1
        grads['W1'] = np.dot(x.T, d_scores)
        grads['b1'] = np.sum(d_hidden, axis=0)

        # add regularization gradient contribution
        dw2_reg = reg * w2
        dw1_reg = reg * w1

        grads['W2'] += dw2_reg
        grads['W1'] += dw1_reg

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, x, y, x_val, y_val,
              learning_rate=1e-3, learning_rate_decay=.95, reg=5e-6,
              n_iters=100, batch_size=200):
        """
        Train this neural network using stochastic gradient descent.

        :param x: A numpy array of shape (N, D) giving training data.
        :param y: A numpy array f shape (N,) giving training labels;
        y[i] = c means that X[i] has label c, where 0 <= c < C.
        :param x_val: A numpy array of shape (N_val, D) giving validation data.
        :param y_val: A numpy array of shape (N_val,) giving validation labels.
        :param learning_rate: Scalar giving learning rate for optimization.
        :param learning_rate_decay: Scalar giving factor used to decay the learning rate after each epoch.
        :param reg: Scalar giving regularization strength.
        :param n_iters: Number of steps to take when optimizing.
        :param batch_size: Number of training examples to use per step.
        """

        n_train = x.shape[0]
        iterations_per_epoch = max(n_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(n_iters):
            #########################################################################
            # TODO: Create a random mini-batch of training data and labels, storing #
            # them in x_batch and y_batch respectively.                             #
            #########################################################################

            # Get training batch data
            rand_idx = np.random.choice(np.arange(n_train), batch_size, replace=True)

            x_batch = x[rand_idx]
            y_batch = y[rand_idx]

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current mini-batch
            loss, grads = self.loss(x_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################

            self.params['W1'] += - learning_rate * grads['W1']
            self.params['b1'] += - learning_rate * grads['b1']
            self.params['W2'] += - learning_rate * grads['W2']
            self.params['b2'] += - learning_rate * grads['b2']

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # logging
            if it % 100 == 0:
                print("[+] Iter %04d, loss : {.:8f}".format(it, np.mean(loss_history)))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = np.mean(self.predict(x_batch) == y_batch)
                val_acc = np.mean(self.predict(x_val) == y_val)

                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, x):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        :param x: A numpy array of shape (N, D) giving N D-dimensional data points to classify.
        :return: A numpy array of shape (N,) giving predicted labels for each of the elements of X.
        For all i, y_pred[i] = c means that X[i] is predicted to have class c, where 0 <= c < C.
        """

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################

        # Neural Net Model
        h1 = x.dot(self.params['W1']) + self.params['b1']
        a1 = np.maximum(0, h1)
        scores = a1.dot(self.params['W2']) + self.params['b2']

        y_predict = np.argmax(scores, axis=1)

        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_predict
