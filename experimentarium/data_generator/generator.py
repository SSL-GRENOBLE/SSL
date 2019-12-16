import os

from . import *
from os.path import normpath, basename

__all__ = ["DataGenerator"]


class DataGenerator(object):
    def __init__(self, data_root: str) -> None:
        self.data_root = data_root

    def __write_file(self, x, y, benchmark):
        folder = benchmark["folder"]
        path = os.path.join(self.data_root, folder)
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path, basename(normpath(path)) + ".data")
        f = open(filename, "w")
        for x_, y_ in zip(x, y):
            f.write(','.join(map(str, x_)) + ',' + str(y_) + '\n')
        f.close()

    def generate(self, benchmark):
        gen_type = benchmark["gen_type"]
        params = benchmark["params"]
        func = globals()[f"generate_{gen_type}"]
        x, y, success = func(params["type"], params["args"])
        if success:
            self.__write_file(x, y, benchmark)
        return success
