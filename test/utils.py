import logging
import os
import sys

from datetime import datetime
from pathlib import Path
from typing import Generator, Iterable, Iterator, Optional, Union

import tqdm


def make_iter(
    obj: Union[Iterable, Generator, Iterator],
    verbose: bool = True,
    desc: Optional[str] = None,
    total: Optional[int] = None,
):
    """Make iterator or tqdm iterator from the given object."""
    if isinstance(obj, Iterable):
        obj_iter = iter(obj)
        total = total or len(obj)
    elif not isinstance(obj, (Iterator, Generator)):
        raise TypeError(f"Cannot make iterator from {type(obj)}.")
    if verbose:
        obj_iter = tqdm.tqdm(obj_iter, total=total, desc=desc)
    return obj_iter


def setup_logger(
    verbose: bool, log: bool, log_root: str = ".ssl_test_logs"
) -> logging.Logger:
    logger = logging.getLogger("ssllogger")
    logger.setLevel(logging.INFO)
    if verbose:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    if log:
        if log_root == ".ssl_test_logs":
            log_root = str(Path(__file__).resolve().parents[0])
        log_root = os.path.join(log_root, ".ssl_test_logs")
        if not os.path.exists(log_root):
            os.mkdir(log_root)
        log_path = os.path.join(
            log_root, f'{datetime.now().strftime("%H-%M-%S_%Y-%m-%d")}'
        )
        open(log_path, "w").close()
        file_handler = logging.FileHandler(log_path)
        logger.addHandler(file_handler)
    return logger
