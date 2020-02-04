import numpy as np

from .defaults import RANDOM_STATE


def generate_spirals(**kwargs):
    params = dict(kwargs.items())
    n_samples = params["n_samples"]
    noise = params.get("noise", 0.5)
    r = np.random.RandomState(RANDOM_STATE)

    n = np.sqrt(r.rand(n_samples, 1)) * 780 * (2 * np.pi) / 360
    dx = -np.cos(n) * n
    dy = np.sin(n) * n

    x = np.vstack(
        (
            np.hstack((dx + r.rand(n_samples, 1) * noise, dy + r.rand(n_samples, 1),)),
            np.hstack(
                (-dx + r.rand(n_samples, 1) * noise, -dy + r.rand(n_samples, 1),)
            ),
        )
    )
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    return x, y


def generate_overlapping_planes(**kwargs):
    size = 2000
    np.random.seed(RANDOM_STATE)

    x_1 = np.random.uniform(-5, 5, size)
    x_2 = np.random.uniform(-5, 5, size)
    y_1 = np.random.uniform(-5, 5, size)
    y_2 = np.random.uniform(-5, 5, size)
    z_1 = 12 * x_1 - 1 * y_1 + np.random.normal(1, 5, size)
    z_2 = -9 * x_2 + 14 * y_2 + np.random.normal(1, 5, size)

    x = np.vstack((
        np.hstack((x_1, x_2)),
        np.hstack((y_1, y_2)),
        np.hstack((z_1, z_2))
    )).reshape(-1, 3)
    y = np.array([0] * size + [1] * size)

    return x, y
