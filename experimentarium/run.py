import argparse
import distutils.util
import json
import importlib.util
import os
import warnings

from configurator import MasterConfiguration, TestRunner
from data_react.webloader import WebDataDownloader

warnings.filterwarnings("ignore")

MODELS = {"sla", "rf", "lda"}

DEFAULT_DATA_ROOT = "../../data/"
DEFAULT_LOG_ROOT = "../../"
DEFAULT_CONFIG_PATH = "../config.py"


def absolutize(path: str) -> str:
    """Make relative path absolute w.r.t. current file."""
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    return path


def extract_configs(args):
    spec = importlib.util.spec_from_file_location("config", args.config_path)
    cmodule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmodule)
    args.configs = cmodule.configs


def check_model(args) -> None:
    if args.model is None:
        raise ValueError("No model given.")
    if args.model not in MODELS:
        raise ValueError(f"Unknown model name.")


def check_benchmarks(args) -> None:
    with open(absolutize("./data_react/dataset2dir.json")) as f:
        dataset2dir = json.load(f)

    unknown = []
    for benchmark in args.benchmarks:
        if benchmark not in dataset2dir:
            unknown.append(benchmark)
    if unknown:
        raise ValueError(f"The following datasets not supported: {' '.join(unknown)}.")

    unknown = []
    web_loader = WebDataDownloader(args.data_root)
    for benchmark in args.benchmarks:
        main_benchmark = dataset2dir[benchmark]
        path = os.path.join(args.data_root, main_benchmark)
        if not os.path.exists(path):
            unknown.append(main_benchmark)
        else:
            for url in web_loader.dir2url[main_benchmark]["urls"]:
                _, filename = os.path.split(url)
                if not os.path.exists(os.path.join(path, filename)):
                    unknown.append(main_benchmark)

    if unknown:
        not_found = []
        for benchmark in unknown:
            if not web_loader.download(benchmark):
                not_found.append(benchmark)
        if not_found:
            raise FileNotFoundError(
                "These datasets not found or could not be downloaded: {}.".format(
                    " ".join(not_found)
                )
            )


def check_config(args) -> None:
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"No config found at {args.config_path}.")
    try:
        extract_configs(args)
    except AttributeError:
        raise AttributeError(f"No proper configuration in {args.config_path}.")

    if args.model not in args.configs:
        raise ValueError(f"No configuration for {args.model}.")


def check_input(args: argparse.Namespace) -> None:
    """Check command line arguments.

    Arguments:
        args: Parsed command line arguments.
    """
    if not os.path.isabs(args.data_root):
        args.data_root = absolutize(args.data_root)

    if not os.path.isabs(args.log_root):
        args.log_root = absolutize(args.log_root)

    if not os.path.isabs(args.config_path):
        args.config_path = absolutize(args.config_path)

    check_model(args)
    check_benchmarks(args)
    check_config(args)

    if args.baseline is None:
        for key in args.configs:
            args.configs[key]["baseline_cls"] = None
            args.configs[key]["baseline_inits"] = None

    if args.lsizes is None:
        args.lsizes = [50]

    args.random_states = list(range(args.n_states))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to train", required=True)
    parser.add_argument("--benchmarks", nargs="+", type=str, required=True)
    parser.add_argument("--data_root", type=str)
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
    parser.add_argument("--lsizes", nargs="*", type=int, help="Labelled sizes to test")
    parser.add_argument(
        "--log_root", type=str, help="Folder to store logs, if logging is on"
    )

    parser.set_defaults(
        n_states=10,
        log="True",
        verbose="True",
        baseline="True",
        check_input="True",
        data_root=DEFAULT_DATA_ROOT,
        log_root=DEFAULT_LOG_ROOT,
        config_path=DEFAULT_CONFIG_PATH,
    )

    args = parser.parse_args()
    check_input(args)

    TestRunner(
        MasterConfiguration(**args.configs[args.model]),
        args.data_root,
        args.random_states,
        args.lsizes,
        args.verbose,
        args.log,
        args.log_root,
    ).run(args.benchmarks)
