from sklearn.datasets import make_blobs

from .defaults import RANDOM_STATE


def generate_gaussian(**params):
    centers = params["centers"]
    std = params["std"]
    n_samples = params["n_samples"]
    n_features = params["n_features"]

    x, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=std,
        n_features=n_features,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return x, y
