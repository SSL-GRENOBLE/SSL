import argparse
import datetime
import distutils.util
import os
import sys

from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.normpath(os.path.join(__file__, "../../../")))  # noqa
from experimentarium.utils import make_iter

sns.set()


DEFAULT_RESULTS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../merged_results")
)

DEFAULT_OUT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../plots")
)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Plotter", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--results-root",
        type=str,
        help="Path to merged results or directory with files to merge",
    )
    parser.add_argument("--out-root", type=str, help="Directory to save plots")
    parser.add_argument(
        "--last",
        type=distutils.util.strtobool,
        help="Whether to take the last created file if --results-root is directory",
    )
    parser.add_argument("--extention", type=str, help="Extention of saved plots")
    parser.add_argument("--metrics", type=str, nargs="+", help="Metrics to plot")
    parser.add_argument(
        "--hard-tresholding",
        type=distutils.util.strtobool,
        help="Whether not to display models with scores less than threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold to set for soft and hard thresholding",
    )
    parser.add_argument(
        "--progress-bar",
        type=distutils.util.strtobool,
        help="Whether to show progress bar over processed benchmarks",
    )

    parser.set_defaults(
        results_root=DEFAULT_RESULTS_ROOT,
        out_root=DEFAULT_OUT_ROOT,
        last="True",
        extention="png",
        metrics=["accuracy", "f1"],
        hard_thresholding="True",
        threshold=0.5,
        progress_bar="True",
    )
    args = parser.parse_args()

    # ======================================================================
    # Plotting things.
    # ======================================================================

    read_csv = partial(pd.read_csv, index_col=False, sep=" ")
    if not os.path.exists(args.results_root):
        raise FileExistsError(f"No such file or folder exists: {args.results_root}.")

    dframes = list()
    if os.path.isdir(args.results_root):
        path_iter = Path(args.results_root).glob("*.csv")
        if args.last:
            dframes.append(read_csv(max(path_iter, key=os.path.getctime)))
        else:
            dframes.extend([read_csv(path) for path in path_iter])
    elif os.path.isfile(args.results_root):
        dframes.append(read_csv(args.results_root))
    else:
        raise ValueError("Given results root is not a file nor a folder.")
    df = pd.concat(dframes)

    out_root = os.path.join(
        args.out_root, datetime.datetime.now().strftime("%H-%M-%S-%Y-%m-%d")
    )

    metrics = list(set(df.columns).intersection(set(args.metrics)))
    if not metrics:
        raise ValueError("No given metric found in merged dataframe.")

    models = tuple(pd.unique(df["model"]))
    markers = tuple(
        marker
        for marker in matplotlib.markers.MarkerStyle.markers
        if marker not in {",", "", " ", "None", None}
    )
    colors = sns.color_palette("muted")

    model2marker = dict(zip(models, markers))
    model2color = dict(zip(models, colors))

    linestyle2ssl = {True: "-", False: "-."}
    marker2ssl = {True: "P", False: "o"}

    # Lsize/Ratio -> Metrics -> Benchmark -> Model -> Score
    diffs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # ======================================================================
    # Plotting joint results.
    # ======================================================================

    for benchmark, benchmark_df in make_iter(
        df.groupby("benchmark"), args.progress_bar, desc="#Benchmarks processed"
    ):
        canvases = create_scaled_canvases(metrics, benchmark, ["Accuracy", "F1 score"])

        ratios = pd.unique(benchmark_df["ratio"])
        lsizes = pd.unique(benchmark_df["lsize"])
        scale = lsizes[0] / ratios[0]

        for model, model_df in benchmark_df.groupby("model"):
            add_to_diff = len(pd.unique(model_df["is_ssl"])) == 2
            is_ssl_groupby = model_df.groupby("is_ssl")
            diff_scores = np.subtract(
                *[np.maximum(_df[metrics].to_numpy(), 0.5) for _, _df in is_ssl_groupby]
            )

            for is_ssl, is_ssl_df in is_ssl_groupby:

                for i, metric in enumerate(metrics):

                    scores = np.maximum(is_ssl_df[metric].to_numpy(), 0.5)
                    mask = scores > args.threshold
                    if args.hard_tresholding and np.any(mask == False):  # noqa
                        mask = np.zeros_like(mask, dtype=np.bool)

                    if np.any(mask == True):  # noqa
                        masked_ratios = ratios[mask]
                        masked_scores = scores[mask]

                        canvases[metric].ax.plot(
                            masked_ratios,
                            masked_scores,
                            label=f"{model.upper()}, {'SSL' if is_ssl else 'Baseline'}",
                            color=model2color[model],
                            marker=marker2ssl[is_ssl],
                            linestyle=linestyle2ssl[is_ssl],
                        )
                        canvases[metric].rescale(scale)

                        for ratio, score, diff_score in zip(
                            masked_ratios, masked_scores, diff_scores[mask, i]
                        ):
                            diffs[ratio][metric][benchmark][model] = diff_score

        benchmark_root = os.path.join(out_root, "joint_plots", benchmark)
        try:
            os.makedirs(benchmark_root)
        except FileExistsError:
            pass

        for metric in metrics:
            canvases[metric].fig.savefig(
                os.path.join(benchmark_root, f"{metric}.{args.extention}")
            )
        plt.close("all")

    # ======================================================================
    # Plotting score difference.
    # ======================================================================
    diff_out_root = os.path.join(out_root, "score_difference")

    # Lsize/Ratio -> Metrics -> Benchmark -> Model -> Score
    def set_label(labelled_models: set, model: str) -> dict:
        if model in labelled_models:
            return dict()
        return dict(label=model)

    def score2edgecolor(score: float) -> str:
        if score == 0:
            return "black"
        elif score > 0:
            return "green"
        return "red"

    handles = dict()

    for lsize, mapping in diffs.items():
        lsize_root = os.path.join(
            diff_out_root, f"lsize_{str(lsize).replace('.', '_')}"
        )
        try:
            os.makedirs(lsize_root)
        except FileExistsError:
            pass
        for metric, mapping_ in mapping.items():
            labelled_models = set()
            fig, ax = plt.subplots(figsize=(13, 10))
            ax.tick_params(axis="x", labelrotation=45)
            ax.set_title(f"{metric} difference at ratio {lsize}")
            ax.set_ylabel(f"Difference")
            for benchmark, mapping__ in mapping_.items():
                for model, score in mapping__.items():
                    ax.scatter(
                        benchmark,
                        score,
                        marker=model2marker[model],
                        color=model2color[model],
                        edgecolors=score2edgecolor(score),
                        **set_label(labelled_models, model),
                        s=200,
                    )
                    if not handles.get(model):
                        handles[model] = plt.scatter(
                            [],
                            [],
                            marker=model2marker[model],
                            color=model2color[model],
                            label=model,
                        )
                    labelled_models.add(model)
            ax.legend(handles=list(handles.values()), numpoints=1)
            fig.savefig(os.path.join(lsize_root, f"{metric}.{args.extention}"))
            plt.close("all")
