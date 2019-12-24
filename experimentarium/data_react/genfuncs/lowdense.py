import numpy as np

from .variables import RANDOM_STATE


def _get_random_data():
    r = np.random.RandomState(RANDOM_STATE)
    x1 = r.randn(2100, 1)
    x2 = r.randn(2100, 1)
    return np.hstack((x1, x2))


def generate_vertical_no_lowdense(**kwargs):
    X = _get_random_data()
    X = X[np.abs(X[:, 0]) >= 0.15]
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype('int')
    return X, y


def generate_no_lowdense(**kwargs):
    X = _get_random_data()
    X = X[np.abs(X[:, 0]) >= 0.15]
    X = X[np.abs(X[:, 1]) >= 0.15]
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype('int')
    return X, y


def generate_lowdense(**kwargs):
    X = _get_random_data()
    X = X[np.abs(X[:, 0]) >= 0.1]
    y = np.logical_or(
        np.logical_and(X[:, 0] > 0, X[:, 1] > 0),
        np.logical_and(X[:, 0] > 0, X[:, 1] < 0)
    ).astype('int')
    return X, y
