import numpy as np


def _generate_with_function(func, **kwargs):
    params = dict(kwargs.items())
    n_samples = params["n_samples"]
    max_value = params["range"]
    noise_limit = params["noise"]

    X = np.empty((n_samples, 2))
    y = np.empty((n_samples,))
    noise = np.random.normal(0, noise_limit, n_samples)

    for i in range(0, n_samples):
        X[i][0] = np.random.uniform(-max_value, max_value)
        X[i][1] = np.random.uniform(-max_value, max_value)
        y[i] = 1 if np.sign(func(X[i][0] + X[i][1]) + noise[i]) > 0 else 0
    return (X, y)


def generate_sign_nonlinear(**kwargs):
    return _generate_with_function(np.sin, **kwargs)


def generate_sign_linear(**kwargs):
    return _generate_with_function(lambda x: x, **kwargs)
