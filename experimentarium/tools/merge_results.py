"""Merges the last produced results."""

import argparse
import datetime
import os

from pathlib import Path

import pandas as pd

DEFAULT_RESULTS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../results")
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ResultMerger")
    parser.add_argument("--results_root", type=str, help="Folder with produced results")
    parser.add_argument("--model", type=str, nargs="+", help="Model results to merge")

    parser.set_defaults(results_root=DEFAULT_RESULTS_ROOT, model=["all"])

    args = parser.parse_args()

    pattern = "*"
    if args.model[0] != "all":
        pattern += f"[{'|'.join(args.model)}]*"

    out_df = None
    for model_root in Path(args.results_root).glob(pattern):
        for bencmark_root in Path(model_root).glob("*"):
            csv_file = list(Path(bencmark_root).glob("*.csv"))[-1]
            df = pd.read_csv(csv_file, sep=" ", index_col=False)
            if out_df is None:
                out_df = df
            else:
                out_df = out_df.append(df, ignore_index=True)

    merge_root = os.path.join(
        Path(args.results_root).resolve().parents[0], "merged_results"
    )
    if not os.path.exists(merge_root):
        os.mkdir(merge_root)
    filename = os.path.join(
        merge_root, f'{datetime.datetime.now().strftime("%H-%M-%S-%Y-%m-%d")}.csv'
    )
    out_df.to_csv(filename, sep=" ", index=False)
