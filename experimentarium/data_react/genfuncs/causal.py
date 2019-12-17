import numpy as np


def generate_sign_nonlinear(**kwargs):
    params = dict(kwargs.items())
    n_samples = params["n_samples"]
    max_value = params["range"]

    x = np.empty((n_samples, 1))
    y = np.empty((n_samples, ))

    for i in range(0, n_samples):
        x[i] = np.random.uniform(-max_value, max_value)
        y[i] = 1 if np.sign(np.sin(x[i])) > 0 else 0
    return (x, y)


def generate_sign_linear(**kwargs):
    params = dict(kwargs.items())
    n_samples = params["n_samples"]
    eps = params["eps"]
    max_value = params["range"]

    x = np.empty((n_samples, 1))
    y = np.empty((n_samples, ))

    for i in range(0, n_samples):
        x[i] = np.random.uniform(-max_value, max_value)
        y[i] = 1 if np.sign(x[i] + eps) > 0 else 0
    return (x, y)
