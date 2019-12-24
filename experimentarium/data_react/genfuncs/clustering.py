from sklearn import datasets

from .defaults import RANDOM_STATE


def generate_moons(**kwargs):
    params = dict(kwargs.items())
    n_samples = params["n_samples"]
    noise = params["noise"]

    x, y = datasets.make_moons(
        n_samples=n_samples, noise=noise, random_state=RANDOM_STATE
    )
    return x, y


def generate_circles(**kwargs):
    params = dict(kwargs.items())
    n_samples = params["n_samples"]
    noise = params["noise"]

    x, y = datasets.make_circles(
        n_samples=n_samples, noise=noise, random_state=RANDOM_STATE
    )
    return x, y


def generate_quadratic(**kwargs):
    params = dict(kwargs.items())
    n_samples = params["n_samples"]
    cluster_std = params["cluster_std"]
    centers = params["centers"]

    x, y = datasets.make_blobs(
        n_samples=n_samples,
        centers=centers,
        shuffle=False,
        cluster_std=cluster_std,
        random_state=RANDOM_STATE,
    )

    y[: n_samples // 2] = 0
    y[n_samples // 2 :] = 1
    return x, y
