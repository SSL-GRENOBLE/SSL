import argparse
import importlib.util
import distutils.util
import os

from pathlib import Path

from tconfigurator import TestConfiguration, Tester
from dataloaders import DATASETS


MODELS = {"sla", "rf", "lda"}


def check_model(args) -> None:
    if args.model is None:
        raise ValueError("No model given.")
    if args.model not in MODELS:
        raise ValueError(f"Unknown model name.")


def check_benchmarks(args):
    if args.benchmarks is None:
        raise ValueError("No benchmarks given.")
    if args.data_root == "../../data":
        args.data_root = str(os.path.join(Path(__file__).resolve().parents[2], "data"))
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"No such folder with data exists: {args.data_root}.")

    unknown = []
    for benchmark in args.benchmarks:
        if benchmark not in DATASETS:
            unknown.append(benchmark)
    if unknown:
        raise ValueError(f"The following datasets not supported: {unknown}.")

    unknown = []
    for benchmark in args.benchmarks:
        if not os.path.exists(os.path.join(args.data_root, DATASETS[benchmark])):
            unknown.append(benchmark)
    if unknown:
        raise FileNotFoundError(
            f"The following datasets not found: {' '.join(unknown)}."
        )


def check_config(args):
    if args.config_path is None:
        raise ValueError("No configuraion path given.")
    try:
        spec = importlib.util.spec_from_file_location("config", args.config_path)
        cmodule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cmodule)
        args.configs = cmodule.configs
    except AttributeError:
        raise AttributeError(f"No proper configuration in {args.config_path}.")

    if args.baseline is None:
        for key in args.configs:
            args.configs[key]["baseline_cls"] = None
            args.configs[key]["baseline_inits"] = None

    if args.model not in args.configs:
        raise ValueError(f"No configuration for {args.model}.")


def check_input(args):
    check_model(args)
    check_benchmarks(args)
    check_config(args)
    if args.lsizes is None:
        args.lsizes = [50]
    args.random_states = list(range(args.n_states))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model(s) to train")
    parser.add_argument("--data_root", type=str)
    parser.add_argument(
        "--baseline",
        help="Whether to train baseline model.",
        type=distutils.util.strtobool,
    )
    parser.add_argument(
        "--log", help="Whether to enable logging.", type=distutils.util.strtobool,
    )
    parser.add_argument(
        "--verbose", help="Whether to print results", type=distutils.util.strtobool,
    )
    parser.add_argument("--benchmarks", nargs="+", type=str)
    parser.add_argument(
        "--config_path", type=str, help="Path to py-file with models' configurations."
    )
    parser.add_argument("--n_states", type=int)
    parser.add_argument("--lsizes", nargs="*", type=int)
    parser.add_argument(
        "--log_root",
        type=str,
        help="Folder to store logs, if logging is on.",
        default=".ssl_test_logs",
    )

    parser.set_defaults(
        log="False",
        verbose="True",
        n_states=10,
        baseline="True",
        data_root="../../data",
    )
    args = parser.parse_args()

    check_input(args)

    test_config = TestConfiguration(**args.configs[args.model])
    tester = Tester(
        test_config,
        args.random_states,
        args.lsizes,
        args.verbose,
        args.log,
        args.log_root,
    )
    tester.run(args.data_root, args.benchmarks)
