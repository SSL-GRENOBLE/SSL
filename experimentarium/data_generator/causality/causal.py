import numpy as np


def _generate_gaussian(args):
    x = []
    y = []
    params = int(len(args)/2)
    for p in range(0, params):
        for y_ in range(0, 500):
            x.append(np.random.normal(args[2*p], args[2*p + 1], 10))
            y.append(0 if y_ > 250 else 1)
    return x, y, True


def generate_causal(type, args):
    func = globals()[f"_generate_{type}"]
    return func(args)
