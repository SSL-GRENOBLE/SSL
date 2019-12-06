import argparse
import json
import os
import sys

from pathlib import Path

# Add `SSL` folder to your PATH.
path = os.path.join(Path(__file__).resolve().parents[2], "experiments")
sys.path.append(path)

from pyshrunner import ShellTestRunner  # noqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--params_path",
        type=str,
        help="Path to json file with shrun parameters.",
        required=True,
    )
    args = parser.parse_args()
    with open(os.path.abspath(args.params_path)) as f:
        params = json.load(f)
    ShellTestRunner().run(**params)
