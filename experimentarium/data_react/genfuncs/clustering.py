import sklearn
from pandas import DataFrame
from sklearn.datasets import samples_generator


def generate_moons(**kwargs):
    params = dict(kwargs.items())
    n_samples = params["n_samples"]
    noise = params["noise"]

    X, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise)
    return (X, y)


def generate_circles(**kwargs):
    params = dict(kwargs.items())
    n_samples = params["n_samples"]
    noise = params["noise"]

    X, y = sklearn.datasets.make_circles(n_samples=n_samples, noise=noise)
    return (X, y)


def generate_quadratic(**kwargs):
    params = dict(kwargs.items())
    n_samples = params["n_samples"]
    cluster_std = params["cluster_std"]
    centers = params["centers"]

    X, y = sklearn.datasets.make_blobs(
        n_samples=n_samples, centers=centers, shuffle=False, cluster_std=cluster_std)

    y[:n_samples // 2] = 0
    y[n_samples // 2:] = 1
    return (X, y)
