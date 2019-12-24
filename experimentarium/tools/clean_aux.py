"""Deletes default created data like results, merged results, plots, logs."""

import argparse
import distutils.util
import os
import shutil


DEFAULT_RESULTS_ROOT = "../../../results"
DEFAULT_MERGE_ROOT = "../../../merged_results"
DEFAULT_PLOTS_ROOT = "../../../plots"
DEFAULT_LOG_ROOT = "../../../logs"
DEFAULT_SYNTHETIC_ROOT = "../../../data/synthetic"
DEFAULT_SYNTHETIC_VISUALISATION_ROOT = "../../../synthetic_data_visualisation"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "CleanAux", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data", type=distutils.util.strtobool, help="Whether to delete synthetic data"
    )

    parser.set_defaults(data="False")

    args = parser.parse_args()
    names = ["RESULTS", "MERGE", "PLOTS", "LOG", "SYNTHETIC_VISUALISATION"]
    if args.data:
        names.append("SYNTHETIC")

    for name in names:
        path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), locals()[f"DEFAULT_{name}_ROOT"])
        )
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
