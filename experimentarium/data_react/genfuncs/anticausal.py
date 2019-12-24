from sklearn.datasets import make_blobs
from .variables import RANDOM_STATE


def generate_gaussian(**kwargs):
    params = dict(kwargs.items())
    centers = params["centers"]
    std = params["std"]
    n_samples = params["n_samples"]
    n_features = params["n_features"]

    random_state = RANDOM_STATE
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=std,
        n_features=n_features,
        shuffle=True,
        random_state=random_state,
    )
    return (X, y)
