import json
import os

from . import rfuncs


__all__ = ["DataReader"]


class DataReader(object):
    def __init__(self, data_root: str) -> None:
        self.data_root = data_root
        with open(os.path.join(os.path.dirname(__file__), "dataset2dir.json")) as f:
            self.dataset2dir = json.load(f)

    def read(self, benchmark: str):
        if benchmark not in self.dataset2dir:
            raise ValueError(f"No support for {benchmark} yet.")
        folder = os.path.join(self.data_root, self.dataset2dir[benchmark])
        func = getattr(rfuncs, f'read_{benchmark}')
        return func(folder)
