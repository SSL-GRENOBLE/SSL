import numpy as np


def generate_gaussian(**kwargs):
    n_samples = 10
    n_features = 2
    n_classes = 2
    return (
        np.ones((n_samples, n_features)),
        np.random.randint(0, n_classes, size=n_samples),
    )
