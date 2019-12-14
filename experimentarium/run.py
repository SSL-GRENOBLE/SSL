import argparse
import datetime
import distutils.util
import importlib.util
import json
import os
import pandas as pd
import warnings

from configurator import MasterConfiguration, TestRunner
from data_react.webloader import WebDataDownloader

warnings.filterwarnings("ignore")


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


def extract_configs(args):
    spec = importlib.util.spec_from_file_location("config", args.config_path)
    cmodule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmodule)
    args.configs = cmodule.configs


def check_config(args) -> None:
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"No config found at {args.config_path}.")
    try:
        extract_configs(args)
    except AttributeError:
        raise AttributeError(f"No proper configuration in {args.config_path}.")


def check_model(args) -> None:
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


def check_benchmarks(args) -> None:
    with open(absolutize("./data_react/dataconfig.json")) as f:
        datacfg = json.load(f)

    data2dir = datacfg["dataset2dir"]
    tag2data = datacfg["tag2data"]

    if args.benchmarks[0] == "all":
        args.benchmarks = list(data2dir)
    else:
        for benchmark in args.benchmarks:
            if benchmark not in data2dir and benchmark not in tag2data:
                raise ValueError("Dataset or tag is not supported: {benchmark}.")

    benchmarks = []
    for benchmark in args.benchmarks:
        if benchmark in tag2data:
            benchmarks.extend(tag2data[benchmark])
        else:
            benchmarks.append(benchmark)
    args.benchmarks = list(set(benchmarks))

    print("Benchmarks parsed: ", args.benchmarks)
    unloaded = []
    web_loader = WebDataDownloader(args.data_root)
    for benchmark in args.benchmarks:
        folder = data2dir[benchmark]
        path = os.path.join(args.data_root, folder)
        if not os.path.exists(path):
            unloaded.append(folder)
        else:
            for url in web_loader.dir2url[folder]["urls"]:
                _, filename = os.path.split(url)
                if not os.path.exists(os.path.join(path, filename)):
                    unloaded.append(folder)

    if unloaded:
        not_found = []
        for benchmark in unloaded:
            if not web_loader.download(benchmark):
                not_found.append(benchmark)
        if not_found:
            raise FileNotFoundError(
                "These datasets not found or could not be downloaded: {}.".format(
                    " ".join(not_found)
                )
            )


def check_input(args: argparse.Namespace) -> None:
    """Check command line arguments.

    Arguments:
        args: Parsed command line arguments.
    """
    for attr in ["data_root", "log_root", "config_path", "results_root"]:
        setattr(args, attr, absolutize(getattr(args, attr)))

    if not os.path.exists(args.results_root):
        os.mkdir(args.results_root)

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

    parser.set_defaults(
        n_states=10,
        log="True",
        verbose="True",
        baseline="True",
        data_root=DEFAULT_DATA_ROOT,
        log_root=DEFAULT_LOG_ROOT,
        config_path=DEFAULT_CONFIG_PATH,
        results_root=DEFAULT_RESULTS_ROOT,
    )

    args = parser.parse_args()
    check_input(args)

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
        )
        runner.run(args.benchmarks)

        model_root = os.path.join(args.results_root, model)
        if not os.path.exists(model_root):
            os.mkdir(model_root)
        filename = datetime.datetime.now().strftime("%H-%M-%S-%Y-%m-%d")

        for benchmark in args.benchmarks:
            benchmark_root = os.path.join(model_root, benchmark)
            if not os.path.exists(benchmark_root):
                os.mkdir(benchmark_root)
            results_path = os.path.join(benchmark_root, filename)
            pd.DataFrame(runner.stats_[benchmark]).to_csv(
                f"{results_path}.csv", sep=" ", index=False
            )
