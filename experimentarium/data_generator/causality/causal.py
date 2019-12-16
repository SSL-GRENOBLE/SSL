def _generate_sign(args):
    pass


def generate_causal(type, args):
    func = globals()[f"_generate_{type}"]
    return func(args)
