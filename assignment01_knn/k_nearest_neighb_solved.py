import numpy as np

"""
Credits: the original code belongs to Stanford CS231n course assignment1. Source link: http://cs231n.github.io/assignments2019/assignment1/
"""


class KNearestNeighbor:

    def __init__(self):
        pass

    def fit(self, X, y):

        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):

        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                diff = X[i] - self.X_train[j]
                dists[i, j] = (diff @ diff.T) ** 0.5

        return dists

    def compute_distances_one_loop(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis=1))

        return dists

    def compute_distances_no_loops(self, X):

        return np.sqrt(
            np.sum(X ** 2, axis=1, keepdims=True) -
            2 * (np.matmul(X, self.X_train.T)) +
            np.sum((self.X_train ** 2), axis=1, keepdims=True).T
        )

    def predict_labels(self, dists, k=1):

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])[:k]]

            unique_labels, counts = np.unique(closest_y, return_counts=True)
            sorted_indices = np.lexsort((counts, unique_labels))
            sorted_labels = unique_labels[sorted_indices]
            sorted_counts = counts[sorted_indices]
            max_count = np.max(sorted_counts)
            most_common_labels = sorted_labels[sorted_counts == max_count]
            y_pred[i] = np.min(most_common_labels)

        return y_pred
