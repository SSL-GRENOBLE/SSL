from sklearn.datasets import make_blobs


def generate_gaussian(**kwargs):
    params = dict(kwargs.items())
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
    )
    return (x, y)
