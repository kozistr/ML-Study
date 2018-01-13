import numpy as np


def l1_loss(data, predict):
    return np.sum(np.abs(predict - data), axis=1)


def l2_loss(data, predict):
    return np.sum(np.power(predict - data, 2), axis=1)


class KNearestNeighbor:
    """
    a k-Nearest-Neighbor with L1, L2 distances
    """

    def __init__(self, distance='L2'):
        """
        :param distance: a type of distance, default L2 distance
        """

        # pre-defined
        self.x_train = None
        self.y_train = None

        self.distance = distance

    def train(self, x, y):
        """
        Train the classifier. For k-nearest neighbors this is just memorizing the training data.

        :param x: A numpy array of shape (num_train, D) containing the training data consisting of num_train samples
        each of dimension D.
        :param y: A numpy array of shape (N,) containing the training labels, where y[i] is the label for X[i].
        :return: None
        """

        self.x_train = x
        self.y_train = y

    def predict(self, x, k=1, num_loops=1):
        """
        Predict labels for test data using this classifier.

        :param x: A numpy array of shape (num_test, D) containing test data consisting of num_test samples
        each of dimension D.
        :param k: The number of nearest neighbors that vote for the predicted labels, default 1
        :param num_loops: Determines which implementation to use to compute distances
        between training points and testing points.
        :return: A numpy array of shape (num_test,) containing predicted labels for the test data,
        where y[i] is the predicted label for the test point X[i].
        """

        if num_loops == 0:
            distances = self.no_loops(x)
        elif num_loops == 1:
            distances = self.one_loops(x)
        elif num_loops == 2:
            distances = self.two_loops(x)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(distances=distances, k=k)

    def no_loops(self, x):
        """
        Compute the distance between each test point in X and each training point in self.X_train
        using no explicit loops.

        :param x: Same as compute_distances_two_loops
        :return: Same as compute_distances_two_loops
        """

        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################

        t = np.sum(np.power(x, 2), axis=1)
        f = np.sum(np.power(self.x_train, 2), axis=1).T
        f = np.tile(f, (500, 5000))

        ft = x.dot(self.x_train.T)

        distance = t + f - 2 * ft

        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################

        return distance

    def one_loops(self, x):
        """
        Compute the distance between each test point in X and each training point in self.X_train using a single loop
        over the test data.

        :param x: Same as compute_distances_two_loops
        :return: Same as compute_distances_two_loops
        """

        num_test = x.shape[0]        # the number of test images
        num_train = self.x_train.shape[0]  # the number of train images

        distance = np.zeros((num_test, num_train))

        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            #######################################################################

            if self.distance == 'L1':
                distance[i, :] = l1_loss(self.x_train, x[i, :])
            elif self.distance == 'L2':
                distance[i, :] = l2_loss(self.x_train, x[i, :])

            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################

        return distance

    def two_loops(self, x):
        """
        Compute the distance between each test point in X and each training point in self.X_train using a nested loop
        over both the training data and the test data.

        :param x: A numpy array of shape (num_test, D) containing test data.
        :return: A numpy array of shape (num_test, num_train) where dists[i, j] is the Euclidean distance
        between the ith test point and the jth training point.
        """

        num_test = x.shape[0]        # the number of test images
        num_train = self.x_train.shape[0]  # the number of train images

        distance = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################

                if self.distance == 'L1':
                    distance[i, j] = l1_loss(self.x_train[j, :], x[i, :])
                elif self.distance == 'L2':
                    distance[i, j] = l2_loss(self.x_train[j, :], x[i, :])

                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################

        return distance

    def predict_labels(self, distances, k):
        """
        Given a matrix of distances between test points and training points, predict a label for each test point.

        :param distances: A numpy array of shape (num_test, num_train) where dists[i, j] gives the distance
        between the ith test point and the jth training point.
        :param k: the number of k, nearest neighbors.
        :return: A numpy array of shape (num_test,) containing predicted labels for the test data,
        where y[i] is the predicted label for the test point X[i].
        """

        num_test = distances.shape[0]
        y_prediction = np.zeros(num_test)

        for i in range(num_test):
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################

            min_y = np.argsort(distances[i, :])[:k]  # find kNN labels

            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################

            # select the most common things ever used
            u, indices = np.unique(min_y, return_inverse=True)
            y_prediction[i] = u[np.argmax(np.bincount(indices))]

            #########################################################################
            #                           END OF YOUR CODE                            #
            #########################################################################

        return y_prediction
