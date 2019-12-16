import json
import os

from . import rfuncs


__all__ = ["DataReader"]


class DataReader(object):
    def __init__(self, data_root: str) -> None:
        self.data_root = data_root
        with open(os.path.join(os.path.dirname(__file__), "dataconfig.json")) as f:
            self.datasets = json.load(f)["datasets"]

    def read(self, benchmark: str):
        if benchmark not in self.datasets:
            raise ValueError(f"No support for {benchmark} yet.")
        folder = os.path.join(
            self.data_root, self.datasets[benchmark]["folder"])
        func = getattr(rfuncs, f"read_{benchmark}")
        return func(folder)
