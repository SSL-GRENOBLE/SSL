import datetime
import json
import os

from functools import partial
from pathlib import Path
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ScaledCanvas(object):
    def __init__(self, fig, ax, ax_scaled) -> None:
        self.fig = fig
        self.ax = ax
        self.ax_scaled = ax_scaled

    def rescale(self, scale: float) -> None:
        xlabels = (np.array(self.ax.get_xticks()) * scale).astype(int)
        xlim = self.ax.get_xlim()
        self.ax_scaled.set_xlim(xlim)
        self.ax_scaled.set_xticklabels(xlabels)
        self.ax_scaled.legend = self.ax.legend
        self.ax.legend()
        self.ax.grid()

    @classmethod
    def create(cls, benchmark: str, ylabel: str) -> "ScaledCanvas":
        fig, ax = plt.subplots(figsize=(13, 10))
        ax_scaled = ax.twiny()
        ax.set_title(f"Dataset: {benchmark}")
        ax.set_ylabel(f"{ylabel}")
        ax.set_xlabel("Ratio")
        ax_scaled.set_xlabel("Lsize")
        ax_scaled.zorder = -1
        return ScaledCanvas(fig, ax, ax_scaled)


def create_scaled_canvases(
    metrics: Tuple, benchmark: str, ylabels: Tuple = None
) -> Dict:
    if ylabels is None:
        ylabels = metrics
    else:
        assert len(metrics) == len(ylabels)
    canvases = dict()
    for metric, ylabel in zip(metrics, ylabels):
        canvases[metric] = ScaledCanvas.create(benchmark, ylabel)
    return canvases


def process_cli_args(args) -> None:
    if not os.path.exists(args.results_root):
        raise FileExistsError(f"No such file or folder exists: {args.results_root}.")

    args.out_root = os.path.join(
        args.out_root, datetime.datetime.now().strftime("%H-%M-%S-%Y-%m-%d")
    )

    args.benchmarks = set(args.benchmarks)

    data_config_path = os.path.normpath(
        os.path.join(__file__, "../../data_react/dataconfig.json")
    )
    with open(data_config_path) as file:
        data_config = json.load(file)

    benchmarks = set()
    if "all" in args.benchmarks:
        benchmarks.update(data_config["datasets"])
    else:
        for name in args.benchmarks:
            benchmarks.update({name} if name in data_config["datasets"] else set())
            benchmarks.update(data_config["tag2data"].get(name, set()))
    args.benchmarks = benchmarks


def load_results(args) -> pd.DataFrame:
    read_csv = partial(pd.read_csv, index_col=False, sep=" ")
    data_frames = list()
    if os.path.isdir(args.results_root):
        path_iter = Path(args.results_root).glob("*.csv")
        if args.last:
            data_frames.append(read_csv(max(path_iter, key=os.path.getctime)))
        else:
            data_frames.extend([read_csv(path) for path in path_iter])
    elif os.path.isfile(args.results_root):
        data_frames.append(read_csv(args.results_root))
    else:
        raise ValueError("Given results root is not a file nor a folder.")
    return pd.concat(data_frames)
