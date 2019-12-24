import numpy as np
from .variables import RANDOM_STATE


def generate_spirals(**kwargs):
    params = dict(kwargs.items())
    n_samples = params["n_samples"]
    noise = params.get("noise", 0.5)
    r = np.random.RandomState(RANDOM_STATE)

    n = np.sqrt(r.rand(n_samples, 1)) * 780 * (2 * np.pi) / 360
    dx = -np.cos(n) * n
    dy = np.sin(n) * n

    X = np.vstack(
        (
            np.hstack((dx + r.rand(n_samples, 1) * noise, dy + r.rand(n_samples, 1),)),
            np.hstack(
                (-dx + r.rand(n_samples, 1) * noise, -dy + r.rand(n_samples, 1),)
            ),
        )
    )
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    return (X, y)
