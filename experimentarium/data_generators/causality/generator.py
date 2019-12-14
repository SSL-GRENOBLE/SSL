from . import causal
from . import anticausal


class CausalityDataGenerator:
    """Generates syntetic data with (anti)causal properties.
    """

    def generate_causal(self, name=None, type="gaussian"):
        func = getattr(causal, f"generate_{type}")
        return func()

    def generate_anticausal(self, name=None, type="sign"):
        func = getattr(anticausal, f"generate_{type}")
        return func()

