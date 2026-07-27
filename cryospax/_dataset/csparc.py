"""cryoJAX compatibility with [CSPARC](https://relion.readthedocs.io/en/release-5.0/)."""

import pathlib
import threading
import warnings
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd

from .._io import read_csparc_file_as_star
from .relion import (
    RELION_DEFAULT_OPTICS_ENTRIES,
    RELION_DEFAULT_PARTICLE_ENTRIES,
    RELION_SUPPORTED_PARTICLE_ENTRIES,
    RelionParticleParameterFile,
    _dict_to_options,
    _select_particles,
    _StarfileData,
    _validate_mode,
    _validate_starfile_data,
)


class CryoSparcParticleParameterFile(RelionParticleParameterFile):
    """CSPARC particle parameter file."""

    def __init__(
        self,
        path_to_csfile: str | pathlib.Path,
        mode: Literal["r", "w"] = "r",
        *,
        # For `mode = 'r'
        path_to_passthrough_csfile: str | pathlib.Path | None = None,
        selection_filter: dict[str, Callable] = {},
        # For `mode = 'w'
        exist_ok: bool = False,
        num_particles: int = 0,
        # For either 'r' or 'w'
        max_optics_groups: int | None = None,
        options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `path_to_csfile`:
            The path to the CryoSPARC `.cs` file. If the path does not exist
            and `mode = 'w'`, an empty dataset will be created.
        - `mode`:
            - If `mode = 'w'`, the dataset is prepared to write new
            *parameters*. This is done by storing an empty dataset in
            `RelionParticleParameterFile.starfile_data`. If a STAR file
            already exists at `path_to_csfile`, set `exist_ok = True`.
            - If `mode = 'r'`, the CryoSPARC file at `path_to_csfile` is read
            and converted into `RelionParticleParameterFile.starfile_data`.
        - `path_to_passthrough_csfile`:
            An optional path to a CryoSPARC passthrough `.cs` file, whose fields
            are merged into those of `path_to_csfile` on the particle `'uid'`.
            Use this when the parameters of interest, such as the CTF or the
            pose, are not present in `path_to_csfile` itself. Only used if
            `mode = 'r'`.
        - `selection_filter`:
            A dictionary used to include only particular dataset elements.
            The keys of this dictionary should be any data entry in the STAR
            file, while the values should be a function that takes in a
            column and returns a boolean mask for the column. For example,
            filter by class using
            `selection_filter["rlnClassNumber"] = lambda x: x == 0`.
        - `exist_ok`:
            If the `path_to_csfile` already exists, if `True` and `mode = 'w'`
            nonetheless stores an empty `RelionParticleParameterFile.starfile_data`.
        - `num_particles`:
            If in `mode = 'w'`, initialize STAR file data to be `num_particles`
            entries. These entries are filled with NaN values and must be set
            via `parameter_file[index] = parameter_info` syntax.
        - `max_optics_groups`:
            The maximum allowed optics group entries in the STAR file. The default
            value of this depends on if `mode = 'r'` or `mode = 'w'`:

            - If `mode = 'r'`:
                By default, `max_optics_groups` is twice the number of optics entries
                in the STAR file.
            - If `mode = 'w'`:
                By default, `max_optics_groups` is equal to `1`.

            !!! info
                This argument can be thought of as the number of allowed calls to
                `parameter_file[...] = parameter_info` or `parameter_file.append(parameter_info)`
                before an error will be thrown. Set this to a large value to be safe.

        - `options`:
            A dictionary of options for modifying the behavior of reading/writing.
            - `'loads_metadata'`:
                If `True`, the resulting dict loads
                the raw metadata from the STAR file that is not otherwise included
                into a `pandas.DataFrame`.
                If this is set to `True`, note that dictionaries cannot pass through
                JIT boundaries without removing the metadata.
                By default, `False`.
            - `'loads_envelope'`:
                If `True`, read in the parameters of the CTF envelope function, i.e.
                "rlnCtfScalefactor" and "rlnCtfBfactor".
                By default, `False`.
            - `'make_image_config'`:
                A function with signature
                `fn(shape, pixel_size, voltage_in_kilovolts)` that
                returns a [`cryojax.simulator.BasicImageConfig`](https://michael-0brien.github.io/cryojax/api/simulator/config/)
                class. Use this argument when it is desired to customize the `image_config`
                returned from this class, i.e.
                `value = parameter_file[0:7]; print(value["image_config"])`.
        """  # noqa: E501
        # Private attributes
        self._options = _dict_to_options(options)
        self._mode = _validate_mode(mode)
        # The STAR file data
        self._path_to_starfile = pathlib.Path(path_to_csfile)
        self._path_to_passthrough_csfile = (
            None
            if path_to_passthrough_csfile is None
            else pathlib.Path(path_to_passthrough_csfile)
        )

        starfile_data, optics_group_info = _load_csfile_data(
            self._path_to_starfile,
            mode,
            self._path_to_passthrough_csfile,
            selection_filter,
            exist_ok,
            num_particles,
            max_optics_groups,
            loads_envelope=self._options["loads_envelope"],
        )
        self._starfile_data = starfile_data
        self._num_optics_groups, self._next_optics_group_index = optics_group_info
        self._lock = threading.Lock()


def _load_csfile_data(
    path_to_csfile: pathlib.Path,
    mode: Literal["r", "w"],
    path_to_passthrough_csfile: pathlib.Path | None,
    selection_filter: dict[str, Callable],
    exist_ok: bool,
    num_particles: int,
    max_optics_groups: int | None,
    loads_envelope: bool,
) -> tuple[_StarfileData, tuple[int, int]]:
    if mode == "r":
        if path_to_csfile.exists():
            if (
                path_to_passthrough_csfile is not None
                and not path_to_passthrough_csfile.exists()
            ):
                raise FileNotFoundError(
                    "Passed a `path_to_passthrough_csfile`, but the CryoSPARC file "
                    f"{str(path_to_passthrough_csfile)} does not exist."
                )
            starfile_data = read_csparc_file_as_star(
                path_to_csfile, path_to_passthrough_csfile
            )
            _validate_starfile_data(starfile_data)
            # Handle particle entries
            if len(selection_filter) > 0:
                starfile_data = _select_particles(starfile_data, selection_filter)
            # Handle optics group entries
            optics_data = starfile_data["optics"]
            num_optics_groups, max_optics_group_index = (
                len(optics_data),
                int(optics_data["rlnOpticsGroup"].max()),
            )
            if pd.isna(max_optics_group_index):
                raise OSError(
                    "Tried to parse the optics group 'rlnOpticsGroup' column to "
                    "retrieve its maximum value, but found that "
                    "it had a NaN value. Make sure that your STAR file is correctly "
                    "formatted."
                )
            if max_optics_groups is None:
                max_optics_groups = num_optics_groups
            starfile_data["optics"] = optics_data.reindex(index=range(max_optics_groups))

            if (
                starfile_data["optics"]["rlnImageSize"].dropna().astype(int) % 2 != 0
            ).any():
                warnings.warn(
                    "Found odd image size in STAR file. We have observed that odd "
                    "images tend to result in bad reconstructions in Relion, probably "
                    "due to wrong angles and shifts as the conventions might differ "
                    "when the image size is odd. Be careful, and make sure your "
                    "pipeline behaves as expected."
                )
        else:
            raise FileNotFoundError(
                f"Set `mode = '{mode}'`, but CryoSPARC file {str(path_to_csfile)} "
                "does not exist. To write a new STAR file, set `mode = 'w'`."
            )
    else:
        if path_to_passthrough_csfile is not None:
            raise ValueError(
                "Initialized a `CryoSparcParticleParameterFile` in `mode = 'w'` "
                "but also passed a `path_to_passthrough_csfile`. Passthrough files "
                "are only read in `mode = 'r'`."
            )
        if path_to_csfile.exists() and not exist_ok:
            raise FileExistsError(
                f"Set `mode = 'w'`, but STAR file {str(path_to_csfile)} already "
                "exists. To read an existing STAR file, set `mode = 'r'` or "
                "to erase an existing STAR file, set `mode = 'w'` and "
                "`exist_ok = True`."
            )
        else:
            if len(selection_filter) == 0:
                num_optics_groups = 0
                max_optics_group_index = 1
                if max_optics_groups is None:
                    max_optics_groups = 1

                relion_particle_entries = (
                    RELION_SUPPORTED_PARTICLE_ENTRIES
                    if loads_envelope
                    else RELION_DEFAULT_PARTICLE_ENTRIES
                )
                starfile_data = dict(
                    optics=pd.DataFrame(
                        data={
                            column: pd.Series(dtype=dtype, index=range(max_optics_groups))
                            for column, dtype in RELION_DEFAULT_OPTICS_ENTRIES
                        }
                    ),
                    particles=pd.DataFrame(
                        data={
                            column: pd.Series(dtype=dtype, index=range(num_particles))
                            for column, dtype in relion_particle_entries
                        }
                    ),
                )
            else:
                raise ValueError(
                    "Initialized a `RelionParticleParameterFile` in `mode = 'w'` "
                    "but also passed a `selection_filter`. Selection is only used "
                    "in `mode = 'r'`."
                )

    return (
        _StarfileData(
            optics=starfile_data["optics"], particles=starfile_data["particles"]
        ),
        (num_optics_groups, max_optics_group_index),
    )
