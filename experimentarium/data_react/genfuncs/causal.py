import numpy as np


def _generate_with_function(func, **kwargs):
    params = dict(kwargs.items())
    n_samples = params["n_samples"]
    max_value = params["range"]
    eps = params["eps"]

    x = np.empty((n_samples, 1))
    y = np.empty((n_samples, ))

    for i in range(0, n_samples):
        x[i] = np.random.uniform(-max_value, max_value)
        y[i] = 1 if np.sign(func(x[i] + eps)) > 0 else 0
    return (x, y)


def generate_sign_nonlinear(**kwargs):
    return _generate_with_function(np.sin, **kwargs)


def generate_sign_linear(**kwargs):
    return _generate_with_function(lambda x: x, **kwargs)
