import json
import logging
import inspect
import os
import warnings

from collections import defaultdict
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from data_react.reader import DataReader
from utils import make_iter, setup_logger


__all__ = ["MasterConfiguration", "TestRunner"]


class MasterConfiguration(object):
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


class TestRunner(object):
    """Test runner for single model configuration."""

    def __init__(
        self,
        configuration: MasterConfiguration,
        data_root: str,
        random_states: List[int],
        lsizes: List[int],
        verbose: bool = True,
        log: bool = False,
        log_root: Optional[str] = None,
        progress_bar: bool = False,
    ) -> None:
        """
            configuration: Model configurations to be tested.
            random_states: Random states.
            lsizes: Labelled examples sizes.
            verbose: Whether to print results.
            log: Whether to log results.
                If True, stores to log_root.
            log_root: Folder to store logs.
                Stores log in `ssl_logs` in `log_root`.
            progress_bar: Whether to show progress bar while execution.
        """
        self.configuration = configuration
        self.data_root = data_root
        self.random_states = random_states
        self.lsizes = lsizes

        self.verbose = verbose
        self.progress_bar = progress_bar
        self.log = log
        self.logger = setup_logger(verbose, log, log_root)

        with open(
            os.path.join(os.path.dirname(__file__), "data_react/dataconfig.json")
        ) as f:
            self.datacfg = json.load(f)

        self.__reader = DataReader(data_root)
        self.__clear_stats()

        self.logger.info("-" * 5 + " Configuration info " + "-" * 5)
        self.logger.info(f"Model: {self.configuration.model_cls}.")
        self.logger.info(f"Baseline: {self.configuration.baseline_cls}.")
        self.logger.info(f"Model params: {self.configuration.model_inits}.")
        self.logger.info(f"Baseline params: {self.configuration.baseline_inits}.")

    def __clear_stats(self):
        self.stats_ = defaultdict(list)
        self.__stats = dict()

    def run(self, benchmarks: Union[str, List[str]]) -> None:
        """
        Arguments:
            root: Path to folder with datasets.
            benchmarks: Names of datasets to test.
        """
        self.__clear_stats()
        if isinstance(benchmarks, str):
            benchmarks = [benchmarks]
        self.logger.info("\n" + "-" * 5 + " Testing " + "-" * 5)
        for benchmark in make_iter(benchmarks, self.progress_bar, "Benchmarks"):
            self.logger.info("\n" + f"... Testing benchmark {benchmark}.")
            self.__stats["benchmark"] = benchmark
            self._test(benchmark)

    def _test(self, benchmark: str) -> None:
        x, y = self.__reader.read(benchmark)
        try:
            n_classes = self.datacfg[benchmark]["n_classes"]
        except KeyError:
            n_classes = len(np.unique(y))
        lsizes = []
        for lsize in self.lsizes:
            if not lsize.is_integer():
                lsize = int(lsize * len(y))
            else:
                lsize = int(lsize)
            if lsize < n_classes:
                lsize = n_classes
                warnings.warn(
                    "Given inappropriate number of labelled samples, "
                    f"changing it for {lsize}."
                )
            lsizes.append(lsize)

        for mode in ["model", "baseline"]:
            if getattr(self.configuration, f"{mode}_cls") is not None:
                for model_config in getattr(self.configuration, f"{mode}_configs"):
                    self._test_config(
                        model_config,
                        x,
                        y,
                        self.random_states,
                        lsizes,
                        self.logger,
                        mode == "model",
                    )

    def _test_config(
        self,
        config: partial,
        x: np.ndarray,
        y: np.ndarray,
        random_states: Union[Iterable[int], int] = 0,
        lsizes: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        is_ssl: bool = True,
    ) -> None:
        """
        Arguments:
            model: Model to test.
            is_ssl: Whether model is semi-supervised.
                Default is True.
        """
        logger.info(
            "\n"
            + f"...... Testing {'semisupervised' if is_ssl else 'supervised'} model:"
        )
        logger.info(f"...... with configuration: {config.keywords or 'default'}.")

        if isinstance(random_states, int):
            random_states = [random_states]

        is_random_state_kw = False
        if "random_state" in inspect.signature(config.func).parameters:
            is_random_state_kw = True

        for lsize in make_iter(lsizes, self.progress_bar, desc="Sizes"):
            scores = defaultdict(list)
            ratio = round(lsize / len(y), 3)

            self.__stats["ratio"] = ratio
            self.__stats["lsize"] = lsize
            for random_state in make_iter(
                random_states, self.progress_bar, desc="\tRandom states"
            ):
                if is_random_state_kw:
                    model = config(random_state=random_state)
                else:
                    model = config()

                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, train_size=lsize, stratify=y, random_state=random_state
                )

                if is_ssl:
                    model.fit(x_train, y_train, x_test)
                else:
                    model.fit(x_train, y_train)
                preds = model.predict(x_test)
                scores["accuracy"].append(accuracy_score(preds, y_test))
                scores["f1"].append(f1_score(preds, y_test))

            self.__stats["usize"] = len(x_test)
            self.__stats["accuracy"] = np.mean(scores["accuracy"]).round(3)
            self.__stats["f1"] = np.mean(scores["f1"]).round(5)
            self.__stats["is_ssl"] = is_ssl

            self.stats_[self.__stats["benchmark"]].append(self.__stats.copy())

            logger.info(
                f"Labeled size = {lsize} (ratio {ratio}). Metrics: "
                "accuracy = {:.5f}., f1 = {:.5f}.".format(
                    self.__stats["accuracy"], self.__stats["f1"]
                )
            )
