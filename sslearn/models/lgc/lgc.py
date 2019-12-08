import numpy as np
from scipy.spatial.distance import cdist


class LGC:
    def __init__(self, alpha=0.8, sigma=0.7, n_iter=100):
        self.alpha = alpha
        self.sigma = sigma
        self.n_iter = n_iter

    def fit(self, ldata: np.ndarray, labels: np.ndarray, *args) -> None:
        """
        Arguments:
            ldata:  Labelled data.
            labels: Labels for labelled data.
            udata:  Unlabelled data.
        """

        self.ldata = ldata
        self.labels = labels

        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        X = np.vstack([self.ldata, data])

        Y = np.zeros((X.shape[0], self.labels.max() + 1), dtype='int')
        Y[np.arange(self.labels.size), self.labels] = 1

        dm = cdist(X, X, 'euclidean')
        rbf = lambda x, sigma: np.exp(-x / (2 * self.sigma ** 2))
        vfunc = np.vectorize(rbf)
        W = vfunc(dm, self.sigma)
        np.fill_diagonal(W, 0)

        D_diag_elements = np.sqrt(W.sum(axis=1))
        D = np.diag(D_diag_elements)
        D_inv = np.diag(1 / D_diag_elements)
        S = np.dot(np.dot(D_inv, W), D)

        F = np.dot(S, Y) * self.alpha + (1 - self.alpha) * Y
        for _ in range(self.n_iter):
            F = np.dot(S, F) * self.alpha + (1 - self.alpha) * Y

        return F.argmax(axis=1)[self.labels.size:]
