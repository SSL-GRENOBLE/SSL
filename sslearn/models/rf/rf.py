from .rf_tree import RFTree

import numpy as np
import random


class RandomForest:
    """RandomForest classifier constructor.

        Attributes:
            random_state - random state
            N - number of trees in random forest
    """

    def __init__(self, random_state, N=100):
        self.N = N
        self.trees = []
        self.oobe = 0
        self.random_state = random_state

    """Prepares data for each tree - random subsampling with return.

        Attributes:
            X - feature matrix of the initial data
            y - label of the initial data
    """

    def _prepare_train_data(self, X, y):
        n_samples = len(y)
        choices = random.choices(range(0, len(y)), k=n_samples)  # indices of samples
        X_i = [X[i] for i in choices]
        y_i = [y[i] for i in choices]
        return X_i, y_i, choices

    """Returns the complement of the indices from X, y.
        
        Attributes:
            X - feature matrix of the initial data
            y - label of the initial data
            indices - indices, that were chosen for the training
    """

    def _prepare_oob_data(self, X, y, indices):
        X_oob = [X[i] for i in range(0, len(X)) if i not in indices]
        y_oob = [y[i] for i in range(0, len(X)) if i not in indices]
        return X_oob, y_oob

    """Generate train and oobe data s.t. len(oobe) != 0.
       
        Attributes:
            X - feature matrix of the initial data
            y - label of the initial data
            X_o - feature matrix to find oobe complement(if different from X)
            y_o - label vector to find oobe complement(if different from y)

        Notes:
            (X_o, y_o) are either fully equal to (X, y) or equal at [0, len(y_o)].
    """

    def _generate_tree_data(self, X, y, X_o=None, y_o=None):
        if X_o is None or y_o is None:
            X_o = X
            y_o = y

        for st in range(0, 10):
            X_i, y_i, idx = self._prepare_train_data(X, y)
            # check that all classes are chosen for training
            if len(set(y_i)) != len(set(y)):
                continue
            X_oob, y_oob = self._prepare_oob_data(X_o, y_o, idx)
            if len(X_oob) != 0:
                return X_i, y_i, X_oob, y_oob
        sys.exit("WARNING: We couldn't find good data fot tree#" + str(i))

    """Fit random forest with initial data and train trees with different data.

        Attributes:
            X - feature matrix of the initial data
            y - label of the initial data
    """

    def fit(self, X, y):
        random.seed(self.random_state)
        self.trees = []
        self.oobe = 0

        for i in range(0, self.N):
            rfTree = RFTree(self.random_state + i)
            X_i, y_i, X_oob, y_oob = self._generate_tree_data(X, y)
            rfTree.fit(X_i, y_i)
            self.trees.append(rfTree)
            self.oobe += rfTree.count_oobe(X_oob, y_oob)
        self.oobe /= self.N

    """Returns Out-Of-The-Bag Error of the random forest, which is the mean of OOBE of all trees.

    """

    def get_oobe(self):
        return self.oobe

    """Predict probabilities of classes for the feature matrix.
       Returns class labels and the probability vector for each sample.

        Attributes:
            X - feature vector to predict distribution
    """

    def predict_proba(self, X):
        p = self.trees[0].predict_proba(X)
        labels = self.trees[0].get_classes()

        for i in range(1, self.N):
            p = np.add(self.trees[i].predict_proba(X), p)
        for i in range(0, len(p)):
            p[i] = [p_ / self.N for p_ in p[i]]

        return p.tolist(), labels

    """Predict labels for X, returns labels y.

        Attributes:
            X - feature matrix to predict labels for it
    """

    def predict(self, X):
        Y = [self.trees[0].predict(X)]
        for i in range(1, self.N):
            Y.append(self.trees[i].predict(X))
        labels = []
        for i in range(0, len(X)):
            labels_i = [y[i] for y in Y]  # labels for X[i] sample
            classes, counts = np.unique(labels_i, return_counts=True)
            l = classes[np.where(counts == max(counts))]
            labels.append(l[0])
        return labels
