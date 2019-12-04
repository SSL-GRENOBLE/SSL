import os

from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_svmlight_file


__all__ = ["DATASETS", "DataReader"]


DATASETS = OrderedDict(
    {
        "banknotes": "01_banknote_authentication",
        "telescope": "08_magic_gamma_telescope",
        "pendigits_4_9": "Pendigits",
        "mnist_4_9": "MNIST"
    }
)


def binarize_pendigits(root: str, *labels):
    if len(labels) != 2:
        raise ValueError(f"Accepts only two numbers, got {len(labels)}.")
    df = load_svmlight_file(root)
    features = df[0].todense().view(type=np.ndarray)
    target = df[1].astype(np.int)
    mask = np.logical_or((target == labels[0]), (target == labels[1]))
    x = features[mask, :]
    y = target[mask]
    y[y == labels[0]] = 0
    y[y == labels[1]] = 1
    return x, y


def binarize_mnist(root: str, *labels):
    if len(labels) != 2:
        raise ValueError(f"Accepts only two numbers, got {len(labels)}.")
    features = np.load(os.path.join("data.npy"))
    target = np.load(os.path.join("targets.npy"))
    mask = np.logical_or((target == labels[0]), (target == labels[1]))
    x = features[mask, :]
    y = target[mask]
    y[y == labels[0]] = 0
    y[y == labels[1]] = 1
    return x, y


def read_pendigits_4_9(root: str):
    path = os.path.join(root, "pendigits")
    return binarize_pendigits(path, 4, 9)


def read_mnist_4_9(root: str):
    return binarize_mnist(root, 4, 9)


def read_telescope(root: str):
    path = str(os.path.join(root, "08_magic_gamma_telescope/magic04.data"))
    df = pd.read_csv(path, header=None)
    x = df.values[:, :-1].astype(float)
    y = LabelEncoder().fit_transform(df.values[:, -1].astype(int))
    return x, y


def read_banknotes(root: str):
    path = os.path.join(root, "data_banknote_authentication.txt")
    df = pd.read_csv(path, sep=",", header=None)
    x = df.values[:, :-1].astype(float)
    y = df.values[:, -1].astype(int)
    return x, y


class DataReader(object):
    def __init__(self, root: str) -> None:
        self.root = root

    def read(self, benchmark: str) -> np.ndarray:
        if benchmark not in DATASETS:
            raise ValueError(f"No support for {benchmark} yet.")
        folder = os.path.join(self.root, DATASETS[benchmark])
        if benchmark == "banknotes":
            return read_banknotes(folder)
        elif benchmark == "telescope":
            return read_telescope(folder)
        elif benchmark == "pendigits_4_9":
            return read_pendigits_4_9(folder)
        elif benchmark == "mnist_4_9":
            return read_mnist_4_9(folder)
