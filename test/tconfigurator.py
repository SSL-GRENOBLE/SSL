import logging
import inspect
import os
import sys

from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataloaders import DataReader
from utils import make_iter


__all__ = ["TestConfiguration", "Tester"]


class TestConfiguration(object):
    def __init__(
        self,
        model_cls: Optional[Any] = None,
        model_inits: Optional[Union[List[Dict], Dict]] = None,
        baseline_cls: Optional[Any] = None,
        baseline_inits: Optional[Union[List[Dict], Dict]] = None,
    ) -> None:
        """
        Arguments:
            model: Semi-supervised model to test.
            model_inits: Semi-supervised model's set of initial parameters to test.
            baseline: Supervised model to test, if any.
            model_inits: Supervised model's set of initial parameters to test, if any.
        """

        if model_cls is None and baseline_cls is None:
            raise ValueError("No models given.")
        if model_inits is None and baseline_inits is None:
            raise ValueError("No parameters for initialisation given.")

        self.model_cls = model_cls
        self.baseline_cls = baseline_cls

        if isinstance(model_inits, dict):
            model_inits = [model_inits]
        if isinstance(baseline_inits, dict):
            baseline_inits = [baseline_inits]
        self.model_inits = model_inits or []
        self.baseline_inits = baseline_inits or []

    @property
    def model_configs(self) -> List:
        if self.model_cls is None:
            return []
        return [partial(self.model_cls, **params) for params in self.model_inits]

    @property
    def baseline_configs(self) -> List:
        if self.baseline_cls is None:
            return []
        return [partial(self.baseline_cls, **params) for params in self.baseline_inits]


def test_model_config(
    model_config: Any,
    x: np.ndarray,
    y: np.ndarray,
    random_states: Union[Iterable[int], int] = 0,
    lsizes: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    ssl: bool = True,
) -> None:
    """
    Arguments:
        model: Model to test.
        ssl: Whether model is semi-supervised.
            Default is True.
    """
    logger.info(
        "\n" + f"...... Testing {'semisupervised' if ssl else 'supervised'} model:"
    )
    logger.info(f"...... with configuration: {model_config.keywords or 'default'}.")

    if isinstance(random_states, int):
        random_states = [random_states]

    verbose = False
    if logger.hasHandlers():
        for handler in logger.handlers:
            if handler.stream == sys.stdout:
                verbose = True
                break

    is_random_state_kw = False
    if "random_state" in inspect.signature(model_config.func).parameters:
        is_random_state_kw = True

    for lsize in make_iter(lsizes, verbose, desc="Sizes"):
        scores = []
        for random_state in make_iter(random_states, verbose, desc="\tRandom states"):
            if is_random_state_kw:
                model = model_config(random_state=random_state)
            else:
                model = model_config()

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, train_size=lsize, stratify=y, random_state=random_state
            )

            if ssl:
                model.fit(x_train, y_train, x_test)
            else:
                model.fit(x_train, y_train)
            preds = model.predict(x_test)
            scores.append(accuracy_score(preds, y_test))

        logger.info(
            "Labeled size = {} (ratio {:.3f}), accuracy = {:.5f}.".format(
                lsize, lsize / len(y_test), np.mean(scores)
            )
        )


class Tester(object):
    def __init__(
        self,
        configuration: TestConfiguration,
        random_states: List[int],
        lsizes: List[int],
        verbose: bool = True,
        log: bool = False,
        log_root: Optional[str] = ".",
    ) -> None:
        """
            configuration: Model configurations to be tested.
            random_states: Random states.
            lsizes: Labelled examples sizes.
            verbose: Whether to print results.
            log: Whether to log results.
                If True, stores to log_root.
            log_root: Folder to store logs.
                Stores log in `.ssl_test_logs` in `log_root`.
        """
        self.configuration = configuration
        self.random_states = random_states
        self.lsizes = lsizes

        if verbose is False and log is False:
            raise ValueError("No results may be reached.")
        self.verbose = verbose
        self.log = log

        logger = logging.getLogger("SSLLogs")
        logger.setLevel(logging.INFO)
        if verbose:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        if log:
            if log_root == ".":
                log_root = str(Path(__file__).resolve().parents[0])
            log_root = os.path.join(log_root, '.ssl_test_logs')
            if not os.path.exists(log_root):
                os.mkdir(log_root)
            log_path = os.path.join(
                log_root, f'{datetime.now().strftime("%H-%M-%S_%Y-%m-%d")}'
            )
            open(log_path, "w").close()
            file_handler = logging.FileHandler(log_path)
            logger.addHandler(file_handler)
        self.logger = logger

        self.logger.info("-" * 5 + " Configuration info " + "-" * 5)
        self.logger.info(f"Model: {self.configuration.model_cls}.")
        self.logger.info(f"Baseline: {self.configuration.baseline_cls}.")
        self.logger.info(f"Model params: {self.configuration.model_inits}.")
        self.logger.info(f"Baseline params: {self.configuration.baseline_inits}.")

    def run(self, root: str, benchmarks: Union[str, List[str]]) -> None:
        """
        Arguments:
            root: Path to folder with datasets.
            benchmarks: Names of datasets to test.
        """
        reader = DataReader(root)
        if isinstance(benchmarks, str):
            benchmarks = [benchmarks]
        self.logger.info("\n" + "-" * 5 + " Testing " + "-" * 5)
        for benchmark in make_iter(benchmarks, self.verbose, "Benchmarks"):
            self.logger.info("\n" + f"... Testing benchmark {benchmark}.")
            x, y = reader.read(benchmark)

            if self.configuration.model_cls is not None:
                for model_config in self.configuration.model_configs:
                    test_model_config(
                        model_config,
                        x,
                        y,
                        self.random_states,
                        self.lsizes,
                        self.logger,
                        True,
                    )

            if self.configuration.baseline_cls is not None:
                for model_config in self.configuration.baseline_configs:
                    test_model_config(
                        model_config,
                        x,
                        y,
                        self.random_states,
                        self.lsizes,
                        self.logger,
                        False,
                    )
