"""Module for running tests via subprocess call."""

import inspect
import os
from pathlib import Path
import platform

from subprocess import call


class ShellKwargConstructor(object):
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict()

        if "log_root" not in kwargs:
            self.kwargs["log_root"] = str(
                Path(inspect.stack()[-1].filename).resolve().parents[0]
            )

        for key, value in kwargs.items():
            self._process(key, value)

    def _process(self, key, value):
        if key == "lsizes":
            self.kwargs[key] = " ".join([str(el) for el in value])
        elif key == "benchmarks":
            self.kwargs[key] = " ".join(value)
        else:
            self.kwargs[key] = value

    def to_shell(self) -> str:
        return " ".join(f"--{key} {value}" for key, value in self.kwargs.items())


class ShellTestRunner(object):
    def __init__(self):
        if platform.system().lower() in ["darwin", "linux"]:
            py_cmd = "python3"
        else:
            py_cmd = "py -3"
        self.py_cmd = py_cmd

    def run(self, **test_params: dict) -> None:
        shell_cmd = "{} {} {}".format(
            self.py_cmd,
            os.path.join(Path(__file__).resolve().parents[0], "shrun.py"),
            ShellKwargConstructor(**test_params).to_shell(),
        )
        call(shell_cmd, shell=True)
