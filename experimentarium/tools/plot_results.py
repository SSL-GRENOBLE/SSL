import argparse
import os
import distutils.util
import sys

from collections import defaultdict

# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from __plot_utils import create_scaled_canvases, load_results, process_cli_args

sys.path.append(os.path.normpath(os.path.join(__file__, "../../../")))  # noqa
from experimentarium.utils import make_iter

sns.set()


DEFAULT_RESULTS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../merged_results")
)

DEFAULT_OUT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../plots")
)


if __name__ == "__main__":
    # ======================================================================
    # Parser setting up.
    # ======================================================================

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
    parser.add_argument(
        "--benchmarks", type=str, nargs="*", help="Which benchmarks to plot",
    )
    parser.add_argument(
        "--joint-plots",
        type=distutils.util.strtobool,
        help="Whether to plot joint plots. False means plotting only sl/ssl plots",
    )
    parser.add_argument(
        "--max-diff-display",
        type=float,
        help="Maximum absolute value on difference score plots",
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
        benchmarks=["all"],
        joint_plots="True",
        max_diff_display=0.075,
    )
    args = parser.parse_args()
    process_cli_args(args)

    # ======================================================================
    # Plotting things.
    # ======================================================================
    df = load_results(args)
    args.benchmarks.intersection_update(pd.unique(df["benchmark"]))
    df = df[df["benchmark"].isin(args.benchmarks)]

    if not args.benchmarks:
        raise ValueError(f"None of provided benchmarks is found in loaded data.")

    metrics = list(set(df.columns).intersection(set(args.metrics)))
    if not metrics:
        raise ValueError("No given metric found in merged dataframe.")

    models = tuple(pd.unique(df["model"]))
    # markers = tuple(
    #     marker
    #     for marker in matplotlib.markers.MarkerStyle.markers
    #     if marker not in {",", "", " ", "None", None}
    # )
    markers = ("s", "o", "v", "X")
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
        df.groupby("benchmark"),
        args.progress_bar,
        desc="#Benchmarks processed/joint plots plotted",
    ):
        if args.joint_plots:
            canvases = create_scaled_canvases(
                metrics, benchmark, ["Accuracy", "F1 score"]
            )

        for model, model_df in benchmark_df.groupby("model"):
            ratios = pd.unique(model_df["ratio"])
            lsizes = pd.unique(model_df["lsize"])
            scale = lsizes[0] / ratios[0]

            # Some model may not have a baseline, but to build difference plot there are
            # needed both.
            add_to_diff = len(pd.unique(model_df["is_ssl"])) == 2
            is_ssl_groupby = model_df.groupby("is_ssl")

            if add_to_diff:
                diff_scores = np.subtract(
                    *[
                        np.maximum(_df[metrics].to_numpy(), 0.5)
                        for _, _df in is_ssl_groupby
                    ]
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

                        if args.joint_plots:
                            canvases[metric].ax.plot(
                                masked_ratios,
                                masked_scores,
                                label="{}, {}".format(
                                    model.upper(), "SSL" if is_ssl else "Baseline"
                                ),
                                color=model2color[model],
                                marker=marker2ssl[is_ssl],
                                linestyle=linestyle2ssl[is_ssl],
                            )
                            canvases[metric].rescale(scale)

                        if add_to_diff:
                            for ratio, score, diff_score in zip(
                                masked_ratios, masked_scores, diff_scores[mask, i]
                            ):
                                diffs[ratio][metric][benchmark][model] = diff_score

        if args.joint_plots:
            benchmark_root = os.path.join(args.out_root, "joint_plots", benchmark)
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
    diff_out_root = os.path.join(args.out_root, "score_difference")
    max_diff_display = args.max_diff_display

    # Lsize/Ratio -> Metrics -> Benchmark -> Model -> Score
    def set_label(labelled_models: set, model: str) -> dict:
        if model in labelled_models:
            return dict()
        return dict(label=model)

    def sign2color(score: float) -> str:
        if score == 0:
            return "black"
        elif score > 0:
            return "green"
        return "red"

    handles = dict()

    for lsize, mapping in make_iter(
        diffs.items(), args.progress_bar, desc="#Difference plots plotted"
    ):
        lsize_root = os.path.join(
            diff_out_root, f"lsize_{str(lsize).replace('.', '_')}"
        )
        try:
            os.makedirs(lsize_root)
        except FileExistsError:
            pass
        for metric, mapping_ in mapping.items():
            labelled_models = set()
            fig, ax = plt.subplots(figsize=(15, 15))
            ax.tick_params(axis="x", labelrotation=45)
            ax.set_title(f"{metric} difference at ratio {lsize}")
            ax.set_ylabel(f"Difference")
            for benchmark, mapping__ in mapping_.items():
                for model, score in mapping__.items():
                    if score <= -max_diff_display:
                        score = -max_diff_display - 1e-3
                    elif score >= max_diff_display:
                        score = max_diff_display + 1e-3
                    ax.scatter(
                        benchmark,
                        score,
                        marker=model2marker[model],
                        color=sign2color(score),
                        edgecolor="white",
                        **set_label(labelled_models, model),
                        s=150,
                        alpha=0.75
                    )
                    if not handles.get(model):
                        handles[model] = plt.scatter(
                            [],
                            [],
                            marker=model2marker[model],
                            color="None",
                            edgecolor="black",
                            label=model,
                        )
                    labelled_models.add(model)
            ax.axhline(max_diff_display, linestyle="--")
            ax.axhline(-max_diff_display, linestyle="--")
            ax.legend(handles=list(handles.values()), numpoints=1)
            fig.savefig(os.path.join(lsize_root, f"{metric}.{args.extention}"))
            plt.close("all")
