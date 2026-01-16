"""
Routines for starfile serialization and deserialization.
"""

import pathlib
from typing import Any, Literal, cast

import pandas as pd
import starfile


def read_starfile(filename: str | pathlib.Path, **kwargs: Any) -> dict[str, pd.DataFrame]:
    """Read a STAR file using
    [`starfile`](https://github.com/teamtomo/starfile).

    **Arguments:**

    - `filename`:
        The path where to read the STAR file. This must include
        a '.star' extension.

    Keyword arguments are passed to `starfile.read`.
    """
    # Make sure filename is valid starfile
    _validate_filename(filename, mode="r")
    # Read starfile
    path_to_filename = pathlib.Path(filename)
    starfile_data = starfile.read(path_to_filename, always_dict=True, **kwargs)
    return cast(dict[str, pd.DataFrame], starfile_data)


def write_starfile(starfile_data, filename: str | pathlib.Path, **kwargs: Any):
    """Write a STAR file using
    [`starfile`](https://github.com/teamtomo/starfile).

    **Arguments:**

    - `starfile_data`:
        A dictionary whose keys are strings and whose entries are
        `pandas.DataFrame`s.
    - `filename`:
        The path where to write the STAR file. This must include
        a '.star' extension.

    Keyword arguments are passed to `starfile.write`.
    """
    # Make sure filename is valid starfile
    _validate_filename(filename, mode="w")
    # Write starfile
    path_to_filename = pathlib.Path(filename)
    return starfile.write(starfile_data, path_to_filename, **kwargs)  # type: ignore


def _validate_filename(filename: str | pathlib.Path, mode: Literal["r", "w"]):
    suffixes = pathlib.Path(filename).suffixes
    if not (len(suffixes) == 1 and suffixes[0] == ".star"):
        raise OSError(
            f"Tried to {('write' if mode == 'w' else 'read')} STAR file, "
            "but the filename does not include a '.star' "
            f"suffix. Got filename '{filename}'."
        )
