import argparse
import json
import os
import shutil
import sys

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("muted")


def _absolutize(path: str) -> str:
    return os.path.join(os.path.dirname(__file__), path)


sys.path.append(_absolutize("../"))

from data_react import DataGenerator  # noqa


CONFIG_PATH = _absolutize("../data_react/dataconfig.json")
DEFAULT_OUT_ROOT = _absolutize("../../../synthetic_data_visualisation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SyntheticDataVisualisator")
    parser.add_argument(
        "--out-root", type=str, help="Root to save visualised synthetic data"
    )
    parser.add_argument("--extention", type=str, help="Extention of saved plots")

    parser.set_defaults(out_root=DEFAULT_OUT_ROOT, extention="png")
    args = parser.parse_args()

    if os.path.exists(args.out_root):
        shutil.rmtree(args.out_root)
    os.makedirs(args.out_root)

    with open(CONFIG_PATH) as file:
        cfgs = json.load(file)["datasets"]

    for benchmark, cfg in cfgs.items():
        is_synthetic = bool(cfg.get("gen_type"))
        is_2d = 2 == cfg.get("n_features")
        if is_synthetic and is_2d:
            x, y = DataGenerator._generate(cfg)
            plt.title(f"Dataset: {benchmark}")
            plt.scatter(x[:, 0], x[:, 1], c=y)
            plt.savefig(os.path.join(args.out_root, f"{benchmark}.{args.extention}"))
            plt.clf()
