import argparse
import datetime
import distutils.util
import importlib.util
import json
import os
import pandas as pd
import warnings
import shutil

from typing import List

from configurator import MasterConfiguration, TestRunner
from data_react import DataGenerator, WebDataDownloader


DEFAULT_DATA_ROOT = "../../data/"
DEFAULT_LOG_ROOT = "../../"
DEFAULT_RESULTS_ROOT = "../../results"
DEFAULT_CONFIG_PATH = "../config.py"


class ConfigError(Exception):
    """Exception when wrong config given."""

    __module__ = Exception.__module__


def absolutize(path: str) -> str:
    """Make relative path absolute w.r.t. current file."""
    if not os.path.isabs(path):
        path = os.path.normpath(
            os.path.abspath(os.path.join(os.path.dirname(__file__), path))
        )
    return path


def extract_configs(args: argparse.Namespace):
    spec = importlib.util.spec_from_file_location("config", args.config_path)
    cmodule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmodule)
    args.configs = cmodule.configs


def check_config(args: argparse.Namespace) -> None:
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"No config found at {args.config_path}.")
    try:
        extract_configs(args)
    except AttributeError:
        raise AttributeError(f"No proper configuration in {args.config_path}.")


def check_model(args: argparse.Namespace) -> None:
    if args.model[0] == "all":
        args.model = list(args.configs)
    else:
        for model in args.model:
            if model not in args.configs:
                raise ValueError(f"No configuration for {model}.")

    for model in args.model:
        if not args.baseline or "baseline_cls" not in args.configs[model]:
            args.configs[model]["baseline_cls"] = None
        for key in ["model_inits", "baseline_inits"]:
            if key not in args.configs[model]:
                args.configs[model][key] = dict()

    for model, params in args.configs.items():
        if "model_cls" not in params and "baseline_cls" not in params:
            raise ConfigError(f"No testing classes are given for model: {model}.")


def parse_benchmarks(args: argparse.Namespace) -> None:
    if args.benchmarks[0] == "all":
        args.benchmarks = list(args.benchmarks)
    else:
        for benchmark in args.benchmarks:
            if benchmark not in args.benchmarks and benchmark not in args.tag2data:
                raise ValueError(f"Dataset or tag is not supported: {benchmark}.")

        benchmarks = set()
        for benchmark in args.benchmarks:
            if benchmark in args.tag2data:
                benchmarks = benchmarks.union(args.tag2data[benchmark])
            else:
                benchmarks.add(benchmark)
        args.benchmarks = list(benchmarks)


def load_benchmarks(args: argparse.Namespace, benchmarks: List) -> None:
    unloaded = []
    web_loader = WebDataDownloader(args.data_root)
    for benchmark in benchmarks:
        folder = args.datasets[benchmark]["folder"]
        path = os.path.join(args.data_root, folder)
        if not os.path.exists(path):
            unloaded.append(folder)
        else:
            for url in web_loader.dir2url[folder]["urls"]:
                _, filename = os.path.split(url)
                if not os.path.exists(os.path.join(path, filename)):
                    unloaded.append(folder)
    web_loader.download(*unloaded)


def generate_benchmarks(args: argparse.Namespace, benchmarks: List) -> None:
    ungenerated = []
    for benchmark in benchmarks:
        folder = args.datasets[benchmark]["folder"]
        path = os.path.join(args.data_root, folder)
        if args.debug and os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path) or not os.listdir(path):
            ungenerated.append(benchmark)
    data_generator = DataGenerator(args.data_root)
    data_generator.generate(*ungenerated)


def check_benchmarks(args: argparse.Namespace) -> None:
    with open(absolutize("./data_react/dataconfig.json")) as f:
        datacfg = json.load(f)
    for attr in ["datasets", "tag2data"]:
        setattr(args, attr, datacfg.get(attr))

    parse_benchmarks(args)

    external = []
    synthetic = []
    for benchmark in args.benchmarks:
        if "gen_type" in args.datasets[benchmark]:
            synthetic.append(benchmark)
        else:
            external.append(benchmark)

    if external:
        load_benchmarks(args, external)
    if synthetic:
        generate_benchmarks(args, synthetic)


def check_input(args: argparse.Namespace) -> None:
    """Check command line arguments.

    Arguments:
        args: Parsed command line arguments.
    """
    for attr in ["data_root", "log_root", "config_path", "results_root"]:
        setattr(args, attr, absolutize(getattr(args, attr)))

    if not os.path.exists(args.results_root):
        os.mkdir(args.results_root)

    if args.ignore_warnings:
        warnings.filterwarnings("ignore")

    check_config(args)
    check_model(args)
    check_benchmarks(args)

    if args.lsizes is None:
        args.lsizes = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5]

    args.random_states = list(range(args.n_states))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, nargs="+", help="Model(s) to train", required=True
    )
    parser.add_argument("--benchmarks", nargs="+", type=str, required=True)
    parser.add_argument("--data_root", type=str, help="Path to folder with dataset.")
    parser.add_argument(
        "--baseline",
        help="Whether to train baseline model",
        type=distutils.util.strtobool,
    )
    parser.add_argument(
        "--log", help="Whether to enable logging", type=distutils.util.strtobool,
    )
    parser.add_argument(
        "--verbose", help="Whether to print results", type=distutils.util.strtobool,
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to py-file with models' configurations"
    )
    parser.add_argument(
        "--n_states", type=int, help="Number of random states to average"
    )
    parser.add_argument(
        "--lsizes",
        nargs="*",
        type=float,
        help="Labelled sizes (int) or ratio (float) to test",
    )
    parser.add_argument(
        "--log_root", type=str, help="Folder to store logs, if logging is on"
    )
    parser.add_argument(
        "--results_root", type=str, help="Folder to store testing result."
    )
    parser.add_argument(
        "--ignore_warnings",
        type=distutils.util.strtobool,
        help="Whether to supress warnings or not",
    )
    parser.add_argument(
        "--debug",
        type=distutils.util.strtobool,
        help="Whether to destroys generated fata",
    )
    parser.add_argument(
        "--progress_bar",
        type=distutils.util.strtobool,
        help="Whether to demonstrate progress bar",
    )
    parser.add_argument(
        "--merge_results",
        type=distutils.util.strtobool,
        help="Whether to merge results after each run",
    )

    parser.set_defaults(
        n_states=10,
        log="False",
        verbose="True",
        baseline="True",
        data_root=DEFAULT_DATA_ROOT,
        log_root=DEFAULT_LOG_ROOT,
        config_path=DEFAULT_CONFIG_PATH,
        results_root=DEFAULT_RESULTS_ROOT,
        ignore_warnings="True",
        debug="True",
        progress_bar="False",
        merge_results="False",
    )

    args = parser.parse_args()
    check_input(args)

    if args.merge_results:
        merged_results = []

    for model in args.model:
        config = args.configs[model]
        runner = TestRunner(
            MasterConfiguration(**config),
            args.data_root,
            args.random_states,
            args.lsizes,
            args.verbose,
            args.log,
            args.log_root,
            args.progress_bar,
        )
        runner.run(args.benchmarks)

        for benchmark in args.benchmarks:
            benchmark_root = os.path.join(args.results_root, benchmark)
            model_root = os.path.join(benchmark_root, model)
            try:
                os.makedirs(model_root)
            except FileExistsError:
                pass
            filename = datetime.datetime.now().strftime("%H-%M-%S-%Y-%m-%d")
            path = os.path.join(model_root, filename)
            df = pd.DataFrame(runner.stats_[benchmark])
            df["model"] = model
            df.to_csv(f"{path}.csv", sep=" ", index=False)
            if args.merge_results:
                merged_results.append(df)

    if args.merge_results:
        merged_root = os.path.join(args.results_root, "merged")
        try:
            os.makedirs(merged_root)
        except FileExistsError:
            pass
        filename = datetime.datetime.now().strftime("%H-%M-%S-%Y-%m-%d")
        path = os.path.join(merged_root, f"{filename}.csv")
        pd.concat(merged_results).to_csv(path, sep=" ", index=False)
