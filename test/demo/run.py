import os
import sys

from pathlib import Path

# Add `SSL` folder to your PATH.
path = os.path.join(Path(__file__).resolve().parents[2], "test")
sys.path.append(path)

from shrunner import ShellTestRunner  # noqa


if __name__ == "__main__":
    # Path to configuration file.
    config_path = os.path.join(Path(__file__).resolve().parents[0], "config.py")

    # Path to folder with data.
    data_root = os.path.join(Path(__file__).resolve().parents[0], "data")

    benchmarks = ["banknotes"]
    lsizes = [50]

    params = {
        "model": "sla",
        "benchmarks": benchmarks,
        "log": True,
        "n_states": 1,
        "lsizes": lsizes,
        "config_path": config_path,
        "data_root": data_root,
    }

    ShellTestRunner().run(**params)
