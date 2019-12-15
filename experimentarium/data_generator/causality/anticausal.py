def _generate_sign(args):
    print("generate_sign anticausal is called", args)
    x = [[1], [2]]
    y = [1]
    return x, y, True


def generate_anticausal(type, args):
    func = globals()[f"_generate_{type}"]
    return func(args)
