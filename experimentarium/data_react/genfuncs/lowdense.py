import numpy as np

from .defaults import RANDOM_STATE


def _get_random_data():
    r = np.random.RandomState(RANDOM_STATE)
    x1 = r.randn(2100, 1)
    x2 = r.randn(2100, 1)
    return np.hstack((x1, x2))


def generate_vertical_no_lowdense(**kwargs):
    x = _get_random_data()
    x = x[np.abs(x[:, 0]) >= 0.15]
    y = np.logical_xor(x[:, 0] > 0, x[:, 1] > 0).astype("int")
    return x, y


def generate_no_lowdense(**kwargs):
    x = _get_random_data()
    x = x[np.abs(x[:, 0]) >= 0.15]
    y = np.logical_or(
        np.logical_and(x[:, 0] > 0, x[:, 1] > 0),
        np.logical_and(x[:, 0] < 0, x[:, 1] > 0)
    ).astype("int")

    return x, y


def generate_lowdense(**kwargs):
    x = _get_random_data()
    x = x[np.abs(x[:, 0]) >= 0.1]
    y = np.logical_or(
        np.logical_and(x[:, 0] > 0, x[:, 1] > 0),
        np.logical_and(x[:, 0] > 0, x[:, 1] < 0),
    ).astype("int")
    return x, y
