import argparse
import datetime
import distutils.util
import os

from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()


DEFAULT_RESULTS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../merged_results")
)

DEFAULT_PLOTS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../plots")
)


class _Canvas(object):
    def __init__(self, fig, ax, ax_scaled) -> None:
        self.fig = fig
        self.ax = ax
        self.ax_scaled = ax_scaled

    @classmethod
    def create(cls, benchmark: str, metrics: str) -> "_Canvas":
        fig, ax = plt.subplots(figsize=(10, 8))
        ax_scaled = ax.twiny()
        ax.set_title(f"Dataset: {benchmark}")
        ax.set_ylabel(f"{metrics}")
        ax.set_xlabel("Ratio")
        ax_scaled.set_xlabel("Lsize")
        ax_scaled.zorder = -1
        return _Canvas(fig, ax, ax_scaled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plotter")
    parser.add_argument(
        "--results-root",
        type=str,
        help="Path to merged results or directory with files to merge",
    )
    parser.add_argument("--plots-root", type=str, help="Directory to save plots")
    parser.add_argument(
        "--last",
        type=distutils.util.strtobool,
        help="Whether to take the last created file if --results-root is directory",
    )
    parser.add_argument("--extention", type=str, help="Extention of saved plots")
    parser.set_defaults(
        results_root=DEFAULT_RESULTS_ROOT,
        plots_root=DEFAULT_PLOTS_ROOT,
        last="True",
        extention="png",
    )
    args = parser.parse_args()

    dframes = list()
    read_csv = partial(pd.read_csv, index_col=False, sep=" ")
    results_root = args.results_root
    if os.path.isdir(results_root):
        path_iter = Path(results_root).glob("*.csv")
        if args.last:
            dframes.append(read_csv(max(path_iter, key=os.path.getctime)))
        else:
            dframes.extend([read_csv(path) for path in path_iter])
    elif os.path.isfile(results_root):
        dframes.append(read_csv(results_root))
    else:
        raise ValueError("Given results root is not a file nor folder.")
    df = pd.concat(dframes)

    out_root = os.path.join(
        args.plots_root, datetime.datetime.now().strftime("%H-%M-%S-%Y-%m-%d")
    )

    colors = sns.color_palette("muted")
    markers = list(matplotlib.markers.MarkerStyle.markers.keys())
    del markers[markers.index(",")]

    do_calc = True
    for _, bench_df in df.groupby("benchmark"):
        benchmark = bench_df.iloc[0]["benchmark"]

        acc_canvas = _Canvas.create(benchmark, "Accuracy")
        f1_canvas = _Canvas.create(benchmark, "F1 score")

        for items in zip(bench_df.groupby(["model"]), colors, markers):
            (_, model_df), color, marker = items
            model = model_df.iloc[0]["model"]

            for _, ssl_df in model_df.groupby("is_ssl"):
                is_ssl = ssl_df.iloc[0]["is_ssl"]

                if do_calc:
                    ratios = ssl_df["ratio"].tolist()

                for canvas, metrics in zip([acc_canvas, f1_canvas], ["accuracy", "f1"]):
                    canvas.ax.plot(
                        ratios,
                        ssl_df[metrics].tolist(),
                        label=f"{model}, {is_ssl}",
                        color=color,
                        marker=marker,
                        linestyle="-" if is_ssl else "-.",
                    )

                    if do_calc:
                        scale = ssl_df.iloc[0]["lsize"] / ratios[0]
                        xlabels = (np.array(canvas.ax.get_xticks()) * scale).astype(int)
                        xlim = canvas.ax.get_xlim()

                    canvas.ax_scaled.set_xlim(xlim)
                    canvas.ax_scaled.set_xticklabels(xlabels)
                    canvas.ax_scaled.legend = canvas.ax.legend
                    canvas.ax.legend()

        do_calc = False
        benchmark_root = os.path.join(out_root, benchmark)
        try:
            os.makedirs(benchmark_root)
        except FileExistsError:
            pass
        acc_canvas.fig.savefig(
            os.path.join(benchmark_root, f"accuracy.{args.extention}")
        )
        f1_canvas.fig.savefig(os.path.join(benchmark_root, f"f1.{args.extention}"))
