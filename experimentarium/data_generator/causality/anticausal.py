def _generate_gaussian(args):
    pass


def generate_anticausal(type, params):
    func = globals()[f"_generate_{type}"]
    return func(args)
