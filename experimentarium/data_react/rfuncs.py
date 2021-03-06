import glob
import gzip
import os

from typing import Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def pair_binarize_multi(x: np.ndarray, y: np.ndarray, label, label_other):
    """Binarizes multiclass dataset w.r.t. `label` and `label_other`.

    Arguments:
        x: Array of features.
        y: Array of labels.
        label: The first label to extract, negative class.
        label_other: The second label to extract, positive class.
    """
    labels = [label, label_other]
    mask = np.logical_or((y == labels[0]), (y == labels[1]))
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


def _read_vehicles(root: str):
    data = glob.glob(os.path.join(root, "xa*.dat"))
    df = pd.concat(
        map(lambda f: pd.read_csv(f, header=None, delim_whitespace=True), data)
    )
    x = df.values[:, :-1].astype(int)
    y = LabelEncoder().fit_transform(df.values[:, -1])
    return x, y


def _read_balance_scale(root: str):
    path = os.path.join(root, "balance-scale.data")
    df = pd.read_csv(path, sep=",", header=None)
    x = df.values[:, 1:].astype(int)
    y = LabelEncoder().fit_transform(df.values[:, 0])
    return x, y


def read_pendigits(root: str):
    path = os.path.join(root, "pendigits_csv.csv")
    df = pd.read_csv(path, sep=",")
    x = df.values[:, :-1].astype(float)
    y = df.values[:, -1].astype(int)
    return x, y


def read_vehicles(root: str):
    return pair_binarize_multi(*_read_vehicles(root), 0, 1)


def read_balance_scale(root: str):
    return pair_binarize_multi(*_read_balance_scale(root), 0, 1)


def read_pendigits_4_9(root: str):
    return pair_binarize_multi(*read_pendigits(root), 4, 9)


def read_mnist_4_9(root: str):
    return pair_binarize_multi(*_read_mnists(root), 4, 9)


def read_fashion_mnist_4_9(root: str):
    return pair_binarize_multi(*_read_mnists(root), 4, 9)


def read_telescope(root: str, n_samples: Optional[float] = None):
    path = str(os.path.join(root, "magic04.data"))
    df = pd.read_csv(path, header=None)
    x = df.values[:, :-1].astype(float)
    y = LabelEncoder().fit_transform(df.values[:, -1])
    if n_samples is not None:
        x, _, y, _ = train_test_split(x, y, train_size=n_samples, stratify=y)
    return x, y


def read_telescope_2000(root: str):
    return read_telescope(root, 2000)


def read_banknotes(root: str):
    path = os.path.join(root, "data_banknote_authentication.txt")
    df = pd.read_csv(path, sep=",", header=None)
    x = df.values[:, :-1].astype(float)
    y = df.values[:, -1].astype(int)
    return x, y


def read_breast_w(root: str):
    path = os.path.join(root, "breast-cancer-wisconsin.data")
    df = pd.read_csv(path, sep=",", header=None)
    df = df.replace("?", np.nan)
    df = df.dropna()
    x = df.values[:, 1:-1].astype(int)
    y = df.values[:, -1].astype(int)
    y[y == 4] = 0
    y[y == 2] = 1
    return x, y


def read_kr_vs_kp(root: str):
    path = os.path.join(root, "kr-vs-kp.data")
    df = pd.read_csv(path, sep=",", header=None)
    df = df.apply(LabelEncoder().fit_transform)
    x = df.values[:, :-1].astype(float)
    y = df.values[:, -1].astype(int)
    return x, y


""" Synthetic data readers """


def _read_synthetic_data(root: str):
    path = os.path.join(root, "data.txt")
    df = pd.read_csv(path, sep=",", header=None)
    x = df.values[:, :-1].astype(float)
    y = df.values[:, -1].astype(int)
    return x, y


def read_ag_dense(root: str):
    return _read_synthetic_data(root)


def read_ag_separable(root: str):
    return _read_synthetic_data(root)


def read_csl_dense(root: str):
    return _read_synthetic_data(root)


def read_csl_sparsed(root: str):
    return _read_synthetic_data(root)


def read_csnl_dense(root: str):
    return _read_synthetic_data(root)


def read_csnl_sparsed(root: str):
    return _read_synthetic_data(root)


def read_moons(root: str):
    return _read_synthetic_data(root)


def read_circles(root: str):
    return _read_synthetic_data(root)


def read_quadratic(root: str):
    return _read_synthetic_data(root)


def read_spirals(root: str):
    return _read_synthetic_data(root)


def read_spirals(root: str):
    return _read_synthetic_data(root)


def read_overlapping_planes(root: str):
    return _read_synthetic_data(root)


def read_no_lowdense(root: str):
    return _read_synthetic_data(root)


def read_lowdense(root: str):
    return _read_synthetic_data(root)
