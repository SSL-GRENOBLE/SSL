import argparse
import os

from functools import partial
from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CSVMerger")
    parser.add_argument(
        "--results-root", type=str, nargs="+", help="Input files or folders"
    )
    parser.add_argument("--out-root", type=str, help="Root to save merged results")

    read_csv = partial(pd.read_csv, index_col=False, sep=" ")
    args = parser.parse_args()
    frames = list()
    for root in args.results_root:
        if os.path.isfile(root):
            frames.append(read_csv(root))
        elif os.path.isdir(root):
            for sub_dir_root in Path(root).glob("**/*.csv"):
                frames.append(read_csv(sub_dir_root))
    pd.concat(frames).to_csv(args.out_root, sep=" ", index=False)
