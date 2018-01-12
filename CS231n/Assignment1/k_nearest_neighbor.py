import numpy as np


def l1_loss(data, predict):
    return np.sum(np.abs(predict - data), axis=1)


def l2_loss(data, predict):
    return np.sum(np.power(predict - data, 2), axis=1)


class KNearestNeighbor:
    """
    a k-Nearest-Neighbor with l1, l2 distances
    """

    def __init__(self, distance='l1'):
        """
        :param distance: a type of distance, default l1 distance
        """

        # pre-defined
        self.x = None
        self.y = None

        self.distance = distance

    def train(self, x, y):
        """
        :param x: x-images (train data)
        :param y: y-labels (train data)
        :return: None
        """

        self.x = x
        self.y = y

    def predict(self, x, k=1, num_loops=1):
        """
        :param x: x-images (test data)
        :param k: the number of nearest neighbors, default 1
        :param num_loops: the number of loops that determines witch method to use to compute 'the distances'
        :return: y (predictions of x)
        """

        distances = 0.
        if num_loops == 0:
            distances = self.no_loops(x)
        elif num_loops == 1:
            distances = self.one_loops(x)
        elif num_loops == 2:
            distances = self.two_loops(x)

        return self.predict_labels(distances=distances, k=k)

    def no_loops(self, x):
        # not sure about this implementation
        t = np.sum(np.power(x, 2), axis=1)
        f = np.sum(np.power(self.x, 2), axis=1).T
        f = np.tile(f, (500, 5000))

        ft = x.dot(self.x.T)

        distance = t + f - 2 * ft

        return distance

    def one_loops(self, x):
        num_test = x.shape[0]        # the number of test images
        num_train = self.x.shape[0]  # the number of train images

        distance = np.zeros((num_test, num_train))

        for i in range(num_test):
            if self.distance == 'l1':
                distance[i, :] = l1_loss(self.x, x[i, :])
            elif self.distance == 'l2':
                distance[i, :] = l2_loss(self.x, x[i, :])

        return distance

    def two_loops(self, x):
        num_test = x.shape[0]        # the number of test images
        num_train = self.x.shape[0]  # the number of train images

        distance = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                if self.distance == 'l1':
                    distance[i, j] = l1_loss(self.x[j, :], x[i, :])
                elif self.distance == 'l2':
                    distance[i, j] = l2_loss(self.x[j, :], x[i, :])

        return distance

    def predict_labels(self, distances, k):
        """
        :param distances: l1, l2 distances
        :param k: the number of nearest neighbors
        :return: predictions
        """

        num_test = distances.shape[0]
        y_prediction = np.zeros(num_test)

        for i in range(num_test):
            min_y = np.argsort(distances[i, :])[:k]  # find kNN labels

            # select the most common things ever used
            u, indices = np.unique(min_y, return_inverse=True)
            y_prediction[i] = u[np.argmax(np.bincount(indices))]

        return y_prediction
