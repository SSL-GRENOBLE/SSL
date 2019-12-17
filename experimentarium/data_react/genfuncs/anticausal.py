import numpy as np


def generate_gaussian(**kwargs):
    params = dict(kwargs.items())
    means = params["means"]
    variances = params["variances"]
    n_samples = params["n_samples"]
    n_classes = len(means)

    x = np.empty((n_samples, 1))
    y = np.empty((n_samples,))
    for i in range(0, n_samples):
        k = np.random.randint(0, n_classes)
        x[i] = np.random.normal(means[k], variances[k])
        y[i] = k
    return (x, y)
