"""Merges the last produced results."""

import argparse
import datetime
import os

from pathlib import Path

import pandas as pd

DEFAULT_RESULTS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../results")
)
DEFAULT_OUT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../merged_results")
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ResultMerger")
    parser.add_argument("--results-root", type=str, help="Folder with produced results")
    parser.add_argument("--out-root", type=str, help="Root to save merged results")
    parser.add_argument("--benchmarks", type=str, nargs="+", help="Benchmarks to merge")
    parser.add_argument("--models", type=str, nargs="+", help="Models to merge")

    parser.set_defaults(
        results_root=DEFAULT_RESULTS_ROOT,
        models=["all"],
        benchmarks=["all"],
        out_root=DEFAULT_OUT_ROOT,
    )

    args = parser.parse_args()

    benchmark_pattern = "*"
    if args.benchmarks[0] != "all":
        benchmark_pattern += f"[{'|'.join(args.benchmarks)}]*"

    models_pattern = "*"
    if args.models[0] != "all":
        models_pattern += f"[{'|'.join(args.models)}]*"

    merged_results = list()
    for benchmark_root in Path(args.results_root).glob(benchmark_pattern):
        for model_root in Path(benchmark_root).glob(models_pattern):
            result_path = max(Path(model_root).glob("*.csv"), key=os.path.getctime)
            merged_results.append(pd.read_csv(result_path, sep=" ", index_col=False))

    if not os.path.exists(args.outы_root):
        os.mkdir(args.out_root)
    path = os.path.join(
        args.out_root, f'{datetime.datetime.now().strftime("%H-%M-%S-%Y-%m-%d")}.csv'
    )
    pd.concat(merged_results).to_csv(path, sep=" ", index=False)
