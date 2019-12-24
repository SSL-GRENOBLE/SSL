import json
import os

from operator import attrgetter
from typing import Tuple

import numpy as np

from . import genfuncs


__all__ = ["DataGenerator"]


class DataGenerator(object):
    """Synthetic data generator."""

    def __init__(self, data_root: str) -> None:
        self.data_root = data_root
        with open(os.path.join(os.path.dirname(__file__), "dataconfig.json")) as file:
            self.datasets = json.load(file)["datasets"]

    def generate(self, *benchmarks) -> None:
        for benchmark in benchmarks:
            cfg = self.datasets[benchmark]
            self._save(
                *self._generate(cfg), os.path.join(self.data_root, cfg["folder"])
            )

    @staticmethod
    def _generate(cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
        gen_type = cfg["gen_type"]
        gen_func = cfg["gen_func"]
        func = attrgetter(f"{gen_type}.generate_{gen_func}")(genfuncs)
        return func(**cfg.get("params", dict()))

    @staticmethod
    def _save(x, y, root: str) -> None:
        if not os.path.exists(root):
            os.makedirs(root)
        filename = os.path.join(root, "data.txt")

        np.savetxt(filename, np.hstack((x, y[:, None])), fmt="%10.5f", delimiter=",")
