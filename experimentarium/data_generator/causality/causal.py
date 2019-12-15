import numpy as np


def _generate_gaussian(args):
    print("generate_gaussian causal is called", type(args))
    x = []
    y = []
    for y_ in range(0, 500):
        x.append(np.random.normal(args[0], args[1], 10))
        y.append(0 if y_ > 250 else 1)

    for y_ in range(0, 500):
        x.append(np.random.normal(args[2], args[3], 10))
        y.append(0 if y_ > 250 else 1)
    return x, y, True


def generate_causal(type, args):
    func = globals()[f"_generate_{type}"]
    return func(args)
