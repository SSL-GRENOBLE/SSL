def _generate_sign(args):
    print("generate_sign anticausal is called", args)


def generate_anticausal(type, args):
    func = globals()[f"_generate_{type}"]
    return func(args)
