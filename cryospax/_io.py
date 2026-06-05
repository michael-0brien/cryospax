"""
Routines for starfile serialization and deserialization.
"""

import pathlib
from typing import Any, Literal, cast

import numpy as np
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
    _validate_filename(filename, mode="r", suffix="star")
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
    _validate_filename(filename, mode="w", suffix="star")
    # Write starfile
    path_to_filename = pathlib.Path(filename)
    return starfile.write(starfile_data, path_to_filename, **kwargs)  # type: ignore


def read_csparc_data(
    filename: pathlib.Path,
    passthrough_filename: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Read a CryoSPARC `.cs` file using `numpy`.

    **Arguments:**

    - `filename`:
        The path where to read the CryoSPARC file. This must include
        a '.cs' extension.
    - `additional_csfiles`:
        Additional CryoSPARC `.cs` files to read. For example, passthrough `.cs` files.

    Each entry in this file is used to populate the columns of a `pandas.DataFrame`.
    """
    _validate_filename(filename, mode="r", suffix="cs")
    if passthrough_filename is not None:
        _validate_filename(passthrough_filename, mode="r", suffix="cs")

    particle_data = _cryosparc_data_to_dataframe(filename)
    if passthrough_filename is not None:
        passthrough_data = _cryosparc_data_to_dataframe(passthrough_filename)
        particle_data = particle_data.merge(passthrough_data, on="uid", how="inner")

    return particle_data


def _cryosparc_data_to_dataframe(
    cryosparc_filename: pathlib.Path,
) -> pd.DataFrame:
    csfile_data = np.load(cryosparc_filename, allow_pickle=True)
    data_entries = [
        csfile_data.dtype.names[i] for i in range(len(csfile_data.dtype.names))
    ]
    csparc_data = pd.DataFrame(
        {
            entry: [csfile_data[j][entry] for j in range(len(csfile_data))]
            for entry in data_entries
        }
    )
    return csparc_data


def _validate_filename(
    filename: str | pathlib.Path, mode: Literal["r", "w"], suffix: Literal["star", "cs"]
):
    suffixes = pathlib.Path(filename).suffixes
    if not (len(suffixes) == 1 and suffixes[0] == f".{suffix}"):
        raise OSError(
            f"Tried to {('write' if mode == 'w' else 'read')} {suffix.upper()} file, "
            f"but the filename does not include a '.{suffix}' "
            f"suffix. Got filename '{filename}'."
        )
