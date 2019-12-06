import gzip
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def binarize_multi(x, y, *labels):
    if len(labels) != 2:
        raise ValueError(f"Accepts only two numbers, got {len(labels)}.")
    mask = np.logical_or((y == labels[0]), (y == y[1]))
    x = x[mask, :]
    y = y[mask]
    y[y == labels[0]] = 0
    y[y == labels[1]] = 1
    return x, y


def _read_mnists(root: str):
    labels_path = os.path.join(root, "t10k-labels-idx1-ubyte.gz")
    images_path = os.path.join(root, "t10k-images-idx3-ubyte.gz")
    with gzip.open(labels_path, "rb") as lbpath:
        y = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, "rb") as imgpath:
        x = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(y), 784
        )
    return x, y


def read_pendigits(root: str):
    path = os.path.join(root, "pendigits_csv.csv")
    df = pd.read_csv(path, sep=",")
    x = df.values[:, :-1].astype(float)
    y = df.values[:, -1].astype(int)
    return x, y


def read_pendigits_4_9(root: str):
    return binarize_multi(*read_pendigits(root), 4, 9)


def read_mnist_4_9(root: str):
    return binarize_multi(*_read_mnists(root), 4, 9)


def read_fashion_mnist_4_9(root: str):
    return binarize_multi(*_read_mnists(root), 4, 9)


def read_telescope(root: str):
    path = str(os.path.join(root, "magic04.data"))
    df = pd.read_csv(path, header=None)
    x = df.values[:, :-1].astype(float)
    y = LabelEncoder().fit_transform(df.values[:, -1])
    return x, y


def read_banknotes(root: str):
    path = os.path.join(root, "data_banknote_authentication.txt")
    df = pd.read_csv(path, sep=",", header=None)
    x = df.values[:, :-1].astype(float)
    y = df.values[:, -1].astype(int)
    return x, y
