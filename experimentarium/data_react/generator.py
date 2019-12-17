import json
import os

from operator import attrgetter

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
            self.__generate(benchmark)

    def __generate(self, benchmark: str) -> None:
        cfg = self.datasets[benchmark]

        gen_type = cfg["gen_type"]
        gen_func = cfg["gen_func"]
        func = attrgetter(f"{gen_type}.generate_{gen_func}")(genfuncs)
        x, y = func(**cfg["params"])

        folder = cfg["folder"]
        path = os.path.join(self.data_root, folder)
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path, "data.txt")

        np.savetxt(filename, np.hstack(
            (x, y[:, None])), fmt="%10.5f", delimiter=",")
