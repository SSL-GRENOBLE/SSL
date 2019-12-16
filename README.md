# SSL

## What is what

* `sslearn` contains models and everything connected with machine learning.
* `experementarium` contains functionality to launch some testing.
* `config.py` parameters of the models that are to be tested.



## Benchmarks

### Adding new ones

One needs to properly provide datareader to `experimentarium/data_react/rfuncs.py`, update `experimentarium/data_react/dataset2dir.json` and `experimentarium/data_react/dir2url.json`.



## Testing

Examine `experementarium/run.py` that supports command line arguments. Than run, supposing your current directory is `SSL`, command

```bash
python3 experementarium/run.py --arg1 value1 --arg2 value2 --arg3 value3
```



 By default it takes data from `../SSL/data` if there is no such folder it tries to download data from the internet.

There is also a possibility to avoid typing arguments to variety command line arguments, just create file,, for instance,`../SSL/run.py` with the following content. 

```python
import sys

repo_path = "SSL"
sys.path.append(repo_path)

from experimentarium.utils import ShellTestRunner  # noqa


if __name__ == "__main__":
    run_root = "SSL/experimentarium"
    run_params = {
        "model": "sla",
        "benchmarks": [
            "pendigits_4_9"
        ],
        "verbose": "True",
        "lsizes": [
            0.1
        ],
        "n_states": 1,
        "log": "True"
    }
    ShellTestRunner().run(run_root, **run_params)
```

The only thing now is to change `run_params` to whatever you need and execute:

```bash
python3 ../SSL/run.py
```

