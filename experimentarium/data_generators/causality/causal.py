def _generate_gaussian(args):
    print("generate_gaussian causal is called", args)


def generate_causal(type, args):
    func = globals()[f"_generate_{type}"]
    return func(args)
