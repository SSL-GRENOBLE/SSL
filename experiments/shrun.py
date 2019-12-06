import argparse
import distutils.util
import json
import importlib.util
import os
import warnings

from pathlib import Path

from tconfigurator import TestConfiguration, Tester
from webdataloader import DataDownloader

warnings.filterwarnings("ignore")

MODELS = {"sla", "rf", "lda"}
DEFAULT_DATA_ROOT = "../../data"


def check_model(args) -> None:
    if args.model is None:
        raise ValueError("No model given.")
    if args.model not in MODELS:
        raise ValueError(f"Unknown model name.")


def check_benchmarks(args):
    if args.benchmarks is None:
        raise ValueError("No benchmarks given.")
    # if not os.path.exists(args.data_root):
    #     raise FileNotFoundError(f"No such folder with data exists: {args.data_root}.")

    with open(
        os.path.join(Path(__file__).resolve().parents[0], "dataset2dataname.json")
    ) as f:
        dataset2dataname = json.load(f)

    unknown = []
    for benchmark in args.benchmarks:
        if benchmark not in dataset2dataname:
            unknown.append(benchmark)
    if unknown:
        raise ValueError(f"The following datasets not supported: {' '.join(unknown)}.")

    unknown = []
    web_loader = DataDownloader(args.data_root)
    for benchmark in args.benchmarks:
        main_benchmark = dataset2dataname[benchmark]
        path = os.path.join(args.data_root, main_benchmark)
        if not os.path.exists(path):
            unknown.append(main_benchmark)
        else:
            for url in web_loader.dataset2url[main_benchmark]["urls"]:
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


def check_config(args):
    if args.config_path is None:
        raise ValueError("No configuraion path given.")
    try:
        if not os.path.isabs(args.config_path):
            args.config_path = str(
                os.path.join(Path(__file__).resolve().parents[0], args.config_path)
            )
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


def properify_input(args):
    if args.data_root == DEFAULT_DATA_ROOT:
        args.data_root = os.path.join(Path(__file__).resolve().parents[2], "data")
    else:
        if not os.path.isabs(args.data_root):
            args.data_root = os.path.join(
                Path(__file__).resolve().parents[0], args.data_root
            )
    if not os.path.isabs(args.log_root):
        curr_dir = os.path.abspath(os.path.dirname(__file__))
        if args.log_root == ".":
            args.log_root = curr_dir
        else:
            args.log_root = os.path.join(curr_dir, args.log_root)
    if args.lsizes is None:
        args.lsizes = [50]
    args.random_states = list(range(args.n_states))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model(s) to train", required=True)
    parser.add_argument("--data_root", type=str, required=True)
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
        "--log_root", type=str, help="Folder to store logs, if logging is on."
    )
    parser.add_argument(
        "--check_input", type=distutils.util.strtobool, help="Whether to check input",
    )

    parser.set_defaults(
        log="False",
        verbose="True",
        n_states=10,
        baseline="True",
        data_root=DEFAULT_DATA_ROOT,
        log_root=".",
        check_input="True",
        config_path="config.py",
    )
    args = parser.parse_args()

    properify_input(args)
    if args.check_input:
        check_input(args)

    test_config = TestConfiguration(**args.configs[args.model])
    tester = Tester(
        test_config,
        args.data_root,
        args.random_states,
        args.lsizes,
        args.verbose,
        args.log,
        args.log_root,
    )
    tester.run(args.benchmarks)
