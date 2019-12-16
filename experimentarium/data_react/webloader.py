import json
import os
import sys
import urllib.request
import warnings

from subprocess import call
from contextlib import contextmanager


__all__ = ["WebDataDownloader"]


def curl(url) -> None:
    """Terminal curl command."""
    call(f"curl --progress-bar -O {str(url)}", shell=True)


@contextmanager
def cd(new_dir: str) -> None:
    """Context manager for changing directory."""
    prev_dir = os.getcwd()
    os.chdir(os.path.expanduser(new_dir))
    try:
        yield
    finally:
        os.chdir(prev_dir)


def is_connected() -> bool:
    """Check if internet connection exists."""
    try:
        urllib.request.urlopen("http://www.google.com/", timeout=20)
        return True
    except urllib.request.URLError:
        return False


class WebDataDownloader(object):
    def __init__(self, root: str) -> None:
        """
        Arguments:
            root: Path to store downloaded data.
                Will be created if not exists.
        """
        self.root = os.path.abspath(root)
        with open(os.path.join(os.path.dirname(__file__), "dir2url.json")) as f:
            self.dir2url = json.load(f)

    def download(self, *benchmarks) -> None:
        for benchmark in benchmarks:
            self.__download(benchmark)

    def __download(self, benchmark: str):
        if not is_connected():
            raise RuntimeError("No internet connection.")
        if benchmark not in self.dir2url:
            raise ValueError(f"Unknown benchmark {benchmark}.")
        if not os.path.exists(self.root):
            os.mkdir(self.root)

        benchmark_root = os.path.join(self.root, benchmark)
        if not os.path.exists(benchmark_root):
            os.mkdir(benchmark_root)

        with cd(benchmark_root):
            for url in self.dir2url[benchmark]["urls"]:
                _, filename = os.path.split(url)
                file_path = os.path.join(benchmark_root, filename)

                if os.path.exists(file_path):
                    warnings.warn(f"File alredy exists, no download.\n")
                else:
                    sys.stdout.write(f"Downloading file to {benchmark_root}.\n")
                    sys.stdout.flush()
                    curl(url)
