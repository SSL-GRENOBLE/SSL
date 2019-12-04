from typing import Generator, Iterable, Iterator, Optional, Union

import tqdm


def make_iter(
    obj: Union[Iterable, Generator, Iterator],
    verbose: bool = True,
    desc: Optional[str] = None,
    total: Optional[int] = None,
):
    """Make iterator or tqdm iterator from the given object."""
    if isinstance(obj, Iterable):
        obj_iter = iter(obj)
        total = total or len(obj)
    elif not isinstance(obj, (Iterator, Generator)):
        raise TypeError(f"Cannot make iterator from {type(obj)}.")
    if verbose:
        obj_iter = tqdm.tqdm(obj_iter, total=total, desc=desc)
    return obj_iter
