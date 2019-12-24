import numpy as np

from .defaults import RANDOM_STATE


def _generate_with_function(func, **params):
    n_samples = params["n_samples"]
    max_value = params["range"]
    noise_limit = params["noise"]

    r = np.random.RandomState(RANDOM_STATE)

    x = np.empty((n_samples, 2))
    y = np.empty((n_samples,))
    noise = r.normal(0, noise_limit, n_samples)

    for i in range(n_samples):
        x[i][0] = r.uniform(-max_value, max_value)
        x[i][1] = r.uniform(-max_value, max_value)
        y[i] = 1 if np.sign(func(x[i][0] + x[i][1]) + noise[0]) > 0 else 0
    return x, y


def generate_sign_nonlinear(**params):
    return _generate_with_function(np.sin, **params)


def generate_sign_linear(**params):
    return _generate_with_function(lambda x: x, **params)
