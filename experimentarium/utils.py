import logging
import os
import platform
import sys

from subprocess import call

from datetime import datetime
from typing import Generator, Iterable, Iterator, Optional, Union

import tqdm


def make_iter(
    obj: Union[Iterable, Generator, Iterator],
    progress_bar: bool = True,
    desc: Optional[str] = None,
    total: Optional[int] = None,
):
    """Make iterator or tqdm iterator from the given object."""
    if isinstance(obj, Iterable):
        obj_iter = iter(obj)
        total = total or len(obj)
    elif not isinstance(obj, (Iterator, Generator)):
        raise TypeError(f"Cannot make iterator from {type(obj)}.")
    if progress_bar:
        obj_iter = tqdm.tqdm(obj_iter, total=total, desc=desc)
    return obj_iter


def setup_logger(
    verbose: bool, log: bool, log_root: Optional[str] = None
) -> logging.Logger:
    """Set up logger.

    Arguments:
        verbose: Whether print to stdout results of logging.
        log: Whether to log to file.
        log_root: Path to folder where to store logs. In this folder there will be
            created folder `ssl_logs` that will contain logs.
    """
    logger = logging.getLogger("ssllogger")
    logger.setLevel(logging.INFO)
    if verbose:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    if log:
        if not os.path.exists(log_root):
            os.mkdir(log_root)
        log_path = os.path.join(
            log_root, f'{datetime.now().strftime("%H-%M-%S-%Y-%m-%d.txt")}'
        )
        open(log_path, "w").close()
        file_handler = logging.FileHandler(log_path)
        logger.addHandler(file_handler)
    return logger


class ShellKwargConstructor(object):
    """Construct experimentarium run arguments from given arguments."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = dict()
        for key, value in kwargs.items():
            self._process(key, value)

    def _process(self, key, value):
        if key == "lsizes":
            if isinstance(value, int):
                value = [value]
            self.kwargs[key] = " ".join([str(el) for el in value])
        elif key in ["model", "benchmarks"]:
            if isinstance(value, str):
                value = [value]
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
            py_cmd = "python"
        self.py_cmd = py_cmd

    def run(self, path: str, **run_params: dict) -> None:
        """Run experimentarium test from python.

        Arguments:
            path: Path to experimentarium folder.
            run_params: Parameters of running.
        """
        shell_cmd = "{} {} {}".format(
            self.py_cmd,
            os.path.join(os.path.abspath(path), "run.py"),
            ShellKwargConstructor(**run_params).to_shell(),
        )
        call(shell_cmd, shell=True)
