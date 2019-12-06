import json
import os
import sys
import urllib.request
import warnings

from pathlib import Path
from subprocess import call
from contextlib import contextmanager


__all__ = ["DataDownloader"]


def curl(url) -> None:
    call(f"curl --progress-bar -O {str(url)}", shell=True)


@contextmanager
def cd(new_dir: str) -> None:
    prev_dir = os.getcwd()
    os.chdir(os.path.expanduser(new_dir))
    try:
        yield
    finally:
        os.chdir(prev_dir)


def is_connected() -> bool:
    try:
        urllib.request.urlopen("http://www.google.com/", timeout=1)
        return True
    except urllib.request.URLError:
        return False


class DataDownloader(object):
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        with open(
            os.path.join(Path(__file__).resolve().parents[0], "dataset2url.json")
        ) as f:
            self.dataset2url = json.load(f)

    def download(self, name: str) -> bool:
        if name not in self.dataset2url:
            raise ValueError(f"Unknown dataset {name}.")
        if not is_connected():
            return False
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        dataset = self.dataset2url[name]
        dataset_root = os.path.join(self.root, name)
        if not os.path.exists(dataset_root):
            os.mkdir(dataset_root)
        with cd(dataset_root):
            for url in dataset["urls"]:
                _, filename = os.path.split(url)
                file_path = os.path.join(dataset_root, filename)
                if os.path.exists(file_path):
                    warnings.warn(f"File alredy exists, no download.\n")
                else:
                    sys.stdout.write(f"Downloading file to {dataset_root}.\n")
                    sys.stdout.flush()
                    curl(url)
        return True
