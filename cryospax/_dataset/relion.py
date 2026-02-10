"""cryoJAX compatibility with [RELION](https://relion.readthedocs.io/en/release-5.0/)."""

import abc
import copy
import pathlib
import threading
import typing
import warnings
from collections.abc import Callable
from typing import Any, Literal, TypedDict
from typing_extensions import NotRequired, Self, override

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd
from cryojax.io import write_image_stack_to_mrc
from cryojax.jax_util import NDArrayLike
from cryojax.ndimage import FourierConstant, FourierGaussian
from cryojax.simulator import (
    AstigmaticCTF,
    BasicImageConfig,
    ContrastTransferTheory,
    EulerAnglePose,
)
from jaxtyping import Array, Float, Int

from .._io import read_starfile, write_starfile
from .._misc import filter_device_get
from .base_dataset import AbstractParticleDataset, AbstractParticleParameterFile


# RELION column entries
RELION_CTF_OPTICS_ENTRIES = [
    ("rlnSphericalAberration", "Float64"),
    ("rlnAmplitudeContrast", "Float64"),
]
RELION_INSTRUMENT_OPTICS_ENTRIES = [
    ("rlnImageSize", "Int64"),
    ("rlnVoltage", "Float64"),
    ("rlnImagePixelSize", "Float64"),
]
RELION_CTF_PARTICLE_ENTRIES = [
    ("rlnDefocusU", "Float64"),
    ("rlnDefocusV", "Float64"),
    ("rlnDefocusAngle", "Float64"),
    ("rlnPhaseShift", "Float64"),
]
RELION_POSE_PARTICLE_ENTRIES = [
    ("rlnOriginXAngst", "Float64"),
    ("rlnOriginYAngst", "Float64"),
    ("rlnAngleRot", "Float64"),
    ("rlnAngleTilt", "Float64"),
    ("rlnAnglePsi", "Float64"),
]
# Default entries for writing
RELION_DEFAULT_OPTICS_ENTRIES = [
    *RELION_INSTRUMENT_OPTICS_ENTRIES,
    *RELION_CTF_OPTICS_ENTRIES,
    ("rlnOpticsGroup", "Int64"),
]
RELION_DEFAULT_PARTICLE_ENTRIES = [
    *RELION_CTF_PARTICLE_ENTRIES,
    *RELION_POSE_PARTICLE_ENTRIES,
    ("rlnOpticsGroup", "Int64"),
]
# Required entries for loading
RELION_REQUIRED_OPTICS_ENTRIES = RELION_DEFAULT_OPTICS_ENTRIES
RELION_REQUIRED_PARTICLE_ENTRIES = [
    *RELION_CTF_PARTICLE_ENTRIES,
    ("rlnOpticsGroup", "Int64"),
]
RELION_SUPPORTED_PARTICLE_ENTRIES = [
    *RELION_REQUIRED_PARTICLE_ENTRIES,
    *RELION_POSE_PARTICLE_ENTRIES,
    ("rlnCtfBfactor", "Float64"),
    ("rlnCtfScalefactor", "Float64"),
]


FloatLike = eqxi.doc_repr(float | Float[np.ndarray, ""], "FloatLike")

MakeImageConfig = Callable[[tuple[int, int], FloatLike, FloatLike], BasicImageConfig]


if hasattr(typing, "GENERATING_DOCUMENTATION"):
    _ParticleParameterInfo = dict[str, Any]  # pyright: ignore[reportAssignmentType]
    _ParticleStackInfo = dict[str, Any]  # pyright: ignore[reportAssignmentType]
    _ParticleParameterLike = dict[str, Any]  # pyright: ignore[reportAssignmentType]
    _ParticleStackLike = dict[str, Any]  # pyright: ignore[reportAssignmentType]
    _Options = dict[str, Any]  # pyright: ignore[reportAssignmentType]
    _StarfileData = dict[str, Any]  # pyright: ignore[reportAssignmentType]
    _MrcfileOptions = dict[str, Any]  # pyright: ignore[reportAssignmentType]

else:

    class _ParticleParameterInfo(TypedDict):
        """Parameters for a particle stack from RELION."""

        image_config: BasicImageConfig
        pose: EulerAnglePose
        transfer_theory: ContrastTransferTheory

        metadata: NotRequired[pd.DataFrame]

    class _ParticleStackInfo(TypedDict):
        """Particle stack info from RELION."""

        images: Float[np.ndarray, "... y_dim x_dim"]
        parameters: NotRequired[_ParticleParameterInfo]

    _ParticleParameterLike = dict[str, Any] | _ParticleParameterInfo
    _ParticleStackLike = dict[str, Any] | _ParticleStackInfo

    class _Options(TypedDict):
        loads_metadata: bool
        loads_envelope: bool
        make_image_config: MakeImageConfig

    class _StarfileData(TypedDict):
        optics: pd.DataFrame
        particles: pd.DataFrame

    class _MrcfileOptions(TypedDict):
        prefix: str
        output_folder: str | pathlib.Path
        n_characters: int
        delimiter: str
        overwrite: bool
        compression: str | None


def _default_make_image_config(shape, pixel_size, voltage_in_kilovolts):
    """Default implementation for generating an `image_config`
    from parameters and the image shape. Additional options passed
    to `BasicImageConfig` may be desired.
    """
    return eqx.tree_at(
        lambda x: (x.pixel_size, x.voltage_in_kilovolts),
        BasicImageConfig(shape, 1.0, 1.0),
        (pixel_size, voltage_in_kilovolts),
    )


default_make_image_config = eqxi.doc_repr(_default_make_image_config, "default_fn")


class AbstractRelionParticleParameterFile(
    AbstractParticleParameterFile[_ParticleParameterInfo, _ParticleParameterLike]
):
    """Abstract class for a RELION particle parameter file.

    This class is mostly useful for wrapping
    the [`cryospax.RelionParticleParameterFile`][] into
    slightly different indexing behavior downstream.
    """

    @property
    @override
    def path_to_output(self) -> pathlib.Path:
        return self.path_to_starfile

    @path_to_output.setter
    @override
    def path_to_output(self, value: str | pathlib.Path):
        self.path_to_starfile = value

    @property
    @abc.abstractmethod
    def path_to_starfile(self) -> pathlib.Path:
        raise NotImplementedError

    @path_to_starfile.setter
    @abc.abstractmethod
    def path_to_starfile(self, value: str | pathlib.Path):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def particle_data(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def optics_data(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def max_optics_groups(self) -> int:
        raise NotImplementedError

    @max_optics_groups.setter
    @abc.abstractmethod
    def max_optics_groups(self, value: int):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def loads_metadata(self) -> bool:
        raise NotImplementedError

    @loads_metadata.setter
    @abc.abstractmethod
    def loads_metadata(self, value: bool):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def loads_envelope(self) -> bool:
        raise NotImplementedError

    @loads_envelope.setter
    @abc.abstractmethod
    def loads_envelope(self, value: bool):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def make_image_config(self) -> MakeImageConfig:
        raise NotImplementedError

    @make_image_config.setter
    @abc.abstractmethod
    def make_image_config(self, value: MakeImageConfig):
        raise NotImplementedError

    def get_metadata(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " _"]
    ) -> pd.DataFrame:
        """Extract the columns at a given `index` *not* included in the cryoJAX
        classes loaded into `parameter_info = parameter_file[index]`.

        This method is for advanced usage; prefer setting
        `parameter_file.loads_metadata = True` and accessing this as
        `parameter_info = parameter_file[...]; metadata = parameter_info["metadata"]`.
        """
        _validate_dataset_index(type(self), index, len(self))
        particle_data_at_index = self.particle_data.iloc[index]
        # ... convert to dataframe for serialization
        if isinstance(particle_data_at_index, pd.Series):
            particle_data_at_index = particle_data_at_index.to_frame().T
        # ... no overlapping keys with loaded pytrees
        redundant_entry_labels, _ = list(zip(*RELION_SUPPORTED_PARTICLE_ENTRIES))
        columns = particle_data_at_index.columns
        remove_columns = [
            column for column in columns if column in redundant_entry_labels
        ]
        metadata = particle_data_at_index.drop(remove_columns, axis="columns")
        return metadata


class RelionParticleParameterFile(AbstractRelionParticleParameterFile):
    """A dataset that wraps a RELION particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html)
    format.

    !!! info "Matching cryoJAX and RELION rotation conventions"

        When simulating images with [`cryojax.simulator.make_image_model`](https://michael-0brien.github.io/cryojax/api/simulator/entry-point/#cryojax.simulator.make_image_model),
        conventions will always match between cryoJAX and RELION.

        However, there are cases where it is necessary to correct for
        the difference with a call to `pose.to_inverse_rotation()`, where
        `pose` is the cryoJAX pose class loaded from the `RelionParticleParameterFile`.

        Whether or not this correction should be made depends on if the pose
        represents a rotation of the *object* or the *frame*. This is encoded
        by the cryoJAX volume representation metadata `volume.rotation_convention`.
        When this is equal to `'frame'` and *not* using `make_image_model`,
        call `pose.to_inverse_rotation()` to match with RELION.
    """  # noqa: E501

    def __init__(
        self,
        path_to_starfile: str | pathlib.Path,
        mode: Literal["r", "w"] = "r",
        *,
        # For `mode = 'r'
        selection_filter: dict[str, Callable] = {},
        # For `mode = 'w'
        exist_ok: bool = False,
        num_particles: int = 0,
        # For either 'r' or 'w'
        max_optics_groups: int | None = None,
        options: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `path_to_starfile`:
            The path to the RELION STAR file. If the path does not exist
            and `mode = 'w'`, an empty dataset will be created.
        - `mode`:
            - If `mode = 'w'`, the dataset is prepared to write new
            *parameters*. This is done by storing an empty dataset in
            `RelionParticleParameterFile.starfile_data`. If a STAR file
            already exists at `path_to_starfile`, set `exist_ok = True`.
            - If `mode = 'r'`, the STAR file at `path_to_starfile` is read
            into `RelionParticleParameterFile.starfile_data`.
        - `selection_filter`:
            A dictionary used to include only particular dataset elements.
            The keys of this dictionary should be any data entry in the STAR
            file, while the values should be a function that takes in a
            column and returns a boolean mask for the column. For example,
            filter by class using
            `selection_filter["rlnClassNumber"] = lambda x: x == 0`.
        - `exist_ok`:
            If the `path_to_starfile` already exists, if `True` and `mode = 'w'`
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
        self._path_to_starfile = pathlib.Path(path_to_starfile)

        starfile_data, optics_group_info = _load_starfile_data(
            self._path_to_starfile,
            mode,
            selection_filter,
            exist_ok,
            num_particles,
            max_optics_groups,
        )
        self._starfile_data = starfile_data
        self._num_optics_groups, self._next_optics_group_index = optics_group_info
        self._lock = threading.Lock()

    @classmethod
    def empty(
        cls: type[Self],
        path_to_starfile: str | pathlib.Path,
        *,
        max_optics_groups: int = 1,
        exist_ok: bool = False,
        num_particles: int = 0,
    ) -> Self:
        """Convenience wrapper for
        [`cryospax.RelionParticleParameterFile.__init__`][] in
        `mode = 'w'` and writing via
        `parameter_file[index] = parameter_info` or
        `parameter_file.append(parameter_info)`.
        """
        return cls(
            path_to_starfile,
            mode="w",
            exist_ok=exist_ok,
            num_particles=num_particles,
            max_optics_groups=max_optics_groups,
        )

    @classmethod
    def load(
        cls: type[Self],
        path_to_starfile: str | pathlib.Path,
        *,
        selection_filter: dict[str, Callable] = {},
        # For writing via `dataset[index] = value`
        max_optics_groups: int | None = None,
        # For loading via `value = dataset[index]`
        loads_metadata: bool = False,
        loads_envelope: bool = False,
        make_image_config: MakeImageConfig = default_make_image_config,
    ) -> Self:
        """Convenience wrapper for
        [`cryospax.RelionParticleParameterFile.__init__`][] in
        `mode = 'r'` and reading via `parameter_info = parameter_file[index]`
        or writing via `parameter_file[index] = parameter_info`.
        """
        return cls(
            path_to_starfile,
            mode="r",
            selection_filter=selection_filter,
            max_optics_groups=max_optics_groups,
            options={
                "loads_metadata": loads_metadata,
                "loads_envelope": loads_envelope,
                "make_image_config": make_image_config,
            },
        )

    @override
    def __getitem__(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " _"]
    ) -> _ParticleParameterInfo:
        """Load STAR file entries with `value = parameter_file[...]` syntax,
        where `value` is a dictionary with keys:

        - `'pose'`:
            The [`cryojax.simulator.EulerAnglePose`](https://michael-0brien.github.io/cryojax/api/simulator/pose/#cryojax.simulator.EulerAnglePose)
        - `'image_config'`:
            The [`cryojax.simulator.BasicImageConfig`](https://michael-0brien.github.io/cryojax/api/simulator/config/#cryojax.simulator.BasicImageConfig)
        - `'transfer_theory'`:
            The [`cryojax.simulator.ContrastTransferTheory`](https://michael-0brien.github.io/cryojax/api/simulator/transfer_theory/#cryojax.simulator.ContrastTransferTheory)
        - `'metadata'`:
            If `loads_metadata = True`, a `pandas.DataFrame` of entries
            *not* used when loading the `pose`, `image_config`, and
            `transfer_theory` (e.g. the 'rlnClassNumber'). Otherwise, this
            key is not included.
        """  # noqa: E501
        # Validate index
        _validate_dataset_index(type(self), index, self.num_particles)
        # ... read particle data at the requested indices
        particle_data = self._starfile_data["particles"]
        particle_data_at_index = particle_data.iloc[index]
        # ... read optics group given the particle data
        optics_data = self._starfile_data["optics"]
        optics_group = _get_optics_group_from_particle_data(
            particle_data_at_index, optics_data
        )
        # Load the image stack and STAR file parameters
        image_config, transfer_theory, pose = _make_pytrees_from_starfile(
            particle_data_at_index,
            optics_group,
            self.loads_envelope,
            self.make_image_config,
        )
        parameter_info = _ParticleParameterInfo(
            image_config=image_config, pose=pose, transfer_theory=transfer_theory
        )
        if self.loads_metadata:
            parameter_info["metadata"] = self.get_metadata(index)

        return parameter_info

    @override
    def __len__(self) -> int:
        """The number of particles in the STAR file. This is simply
        an alias to [`cryospax.RelionParticleParameterFile.num_particles`][].
        """
        return self.num_particles

    @override
    def __setitem__(
        self,
        index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " _"],
        value: _ParticleParameterLike,
    ):
        """Set STAR file entries with `parameter_file[...] = value` syntax,
        where `value` is a dictionary with keys:

        - `'pose'`:
            The [`cryojax.simulator.EulerAnglePose`](https://michael-0brien.github.io/cryojax/api/simulator/pose/#cryojax.simulator.EulerAnglePose)
        - `'image_config'`:
            The [`cryojax.simulator.BasicImageConfig`](https://michael-0brien.github.io/cryojax/api/simulator/config/#cryojax.simulator.BasicImageConfig)
        - `'transfer_theory'`:
            The [`cryojax.simulator.ContrastTransferTheory`](https://michael-0brien.github.io/cryojax/api/simulator/transfer_theory/#cryojax.simulator.ContrastTransferTheory)
        - `'metadata'`:
            A `pandas.DataFrame` used to write custom entries to the STAR
            file. `len(metadata)` should be equal to the number of particles,
            and `metadata.columns` should be a subset of columns that already
            exist in `parameter_file.particle_data` but do not include entries
            filled by the `image_config`, `transfer_theory`, and `pose`.

        All keys are optional, and unless `image_config` is not given,
        `parameter_file[...] = value` creates a new optics group.

        !!! warning

            Setting `parameter_file[...] = value` where `value` has a
            `transfer_theory` but not an `image_config` may result in surprising
            behavior: a new optics group will not be created, even though
            'rlnSphericalAberration' and 'rlnAmplitudeContrast' are parameters
            stored in the `transfer_theory`.

            This is because there is not enough information to create a new
            optics group if both are not given, and an error is not thrown so that patterns like

            ```python
            value = {"transfer_theory": cxs.ContrastTransferTheory(...), "pose": cxs.EulerAnglePose}(...)
            parameter_file[index] = value
            ```

            can be used to set particle entries.
        """  # noqa: E501
        # Make sure index is valid
        try:
            _validate_dataset_index(type(self), index, self.num_particles)
        except IndexError as err:
            raise RuntimeError(
                "Found an invalid index when writing parameters via "
                "`parameter_file[index] = ...` "
                "syntax. If you are creating a new STAR file, a common reason "
                "for this is exceeding the `num_particles` set during initialization, "
                "e.g. `RelionParticleParameterFile.empty(..., num_particles=...)`."
            ) from err
        # ... also, the parameters too
        _validate_parameters(value, force_keys=False)
        # Grab the current optics and particle data
        particle_data, optics_data = (
            self._starfile_data["particles"],
            self._starfile_data["optics"],
        )
        if {"image_config"}.issubset(value):
            # Make and set the new optics data
            optics_group_index, optics_array_index = self._increment_optics_group()
            optics_data_for_update = _parameters_to_optics_data(value, optics_group_index)
            optics_data.loc[
                optics_data.index[np.atleast_1d(optics_array_index)],
                optics_data_for_update.columns,
            ] = optics_data_for_update.values
        else:
            optics_group_index = None
        # Make and set the new particle data
        # ... make
        particle_data_for_update = _parameters_to_particle_data(value, optics_group_index)
        # ... make sure there are no columns that are not recognized
        unrecognized_columns = list(
            set(particle_data_for_update.columns) - set(particle_data.columns)
        )
        if len(unrecognized_columns) > 0:
            raise ValueError(
                "When writing parameters via `parameter_file[index] = value`, "
                "found columns in `value['metadata']` that do not exist in "
                "`parameter_file.particle_data`. "
                f"These unrecognized column(s) were: {unrecognized_columns}. "
                "If you would like to write a custom column 'foo', first instantiate "
                "this column directly as "
                "`parameter_file.particle_data['foo'] = pandas.Series(dtype=...)`."
            )
        # ... set
        if isinstance(index, (int, np.ndarray)):
            index = np.atleast_1d(index)
        particle_data.loc[
            particle_data.index[index], particle_data_for_update.columns
        ] = particle_data_for_update.values
        self._starfile_data["optics"] = optics_data
        self._starfile_data["particles"] = particle_data

    @override
    def append(self, value: _ParticleParameterLike):
        """Add an entry or entries to the STAR file with
        `parameter_file.append(value)` syntax, where `value` is
        a dictionary with keys:

        - `'pose'`:
            The [`cryojax.simulator.EulerAnglePose`](https://michael-0brien.github.io/cryojax/api/simulator/pose/#cryojax.simulator.EulerAnglePose)
            This key is required.
        - `'image_config'`:
            The [`cryojax.simulator.BasicImageConfig`](https://michael-0brien.github.io/cryojax/api/simulator/config/#cryojax.simulator.BasicImageConfig)
            This key is required.
        - `'transfer_theory'`:
            The [`cryojax.simulator.ContrastTransferTheory`](https://michael-0brien.github.io/cryojax/api/simulator/transfer_theory/#cryojax.simulator.ContrastTransferTheory)
            This key is required.
        - `'metadata'`:
            A `pandas.DataFrame` used to write custom entries to the STAR
            file. `len(metadata)` should be equal to the number of particles,
            and `metadata.columns` should be a subset of columns that already
            exist in `parameter_file.particle_data` but do not include entries
            filled by the `image_config`, `transfer_theory`, and `pose`.
            This key is optional.

        A new optics group will be created on each call to
        `parameter_file.append(value)`.
        """
        # Make sure parameters are valid
        _validate_parameters(value, force_keys=True)
        # Make new optics group
        optics_group_index, optics_array_index = self._increment_optics_group()
        optics_data, optics_data_for_update = (
            self._starfile_data["optics"],
            _parameters_to_optics_data(value, optics_group_index),
        )
        optics_data.loc[
            optics_data.index[np.atleast_1d(optics_array_index)],
            optics_data_for_update.columns,
        ] = optics_data_for_update.values
        # Make new particle entries
        particle_data, particle_data_to_append = (
            self._starfile_data["particles"],
            _parameters_to_particle_data(value, optics_group_index),
        )
        # Concatenate and set new entries
        particle_data = (
            pd.concat([particle_data, particle_data_to_append], ignore_index=True)
            if len(particle_data) > 0
            else particle_data_to_append
        )
        self._starfile_data["optics"] = optics_data
        self._starfile_data["particles"] = particle_data

    @override
    def save(
        self,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ):
        """Save the STAR file at the current `parameter_file.path_to_starfile`."""
        path_to_starfile = self.path_to_starfile
        path_exists = path_to_starfile.exists()
        if path_exists and not overwrite:
            raise FileExistsError(
                f"Tried saving STAR file, but file {str(path_to_starfile)} "
                "already exists. To overwrite existing STAR file, set "
                f"`{type(self).__name__}.overwrite = True`."
            )
        else:
            # Make directory if it does not exist
            if not path_to_starfile.parent.exists():
                path_to_starfile.parent.mkdir(parents=True)
            starfile_data = _StarfileData(
                particles=self.particle_data,
                optics=self.optics_data.iloc[: self.num_optics_groups],
            )
            write_starfile(starfile_data, path_to_starfile, **kwargs)

    @property
    @override
    def path_to_starfile(self) -> pathlib.Path:
        """The path to the STAR file. This is an alias to
        `parameter_file.path_to_output`.

        !!! example "Modify and write a new STAR file"
            ```python
            parameter_file = RelionParticleParameterFile(
                path_to_starfile="./path/to/particles.star", mode="r"
            )
            updated_parameters = ...
            parameter_file[...] = updated_parameters
            parameter_file.path_to_starfile = "./path/to/new/particles.star"
            parameter_file.save()
            ```
        """
        return self._path_to_starfile

    @path_to_starfile.setter
    @override
    def path_to_starfile(self, value: str | pathlib.Path):
        self._path_to_starfile = pathlib.Path(value)

    @property
    def mode(self) -> Literal["r", "w"]:
        """Whether or not the `parameter_file` was
        instantiated in reading ('r') or writing ('w') mode.

        This cannot be modified after initialization.
        """
        return self._mode  # type: ignore

    @property
    @override
    def particle_data(self) -> pd.DataFrame:
        """The `pandas.DataFrame` of particle STAR file entries."""
        return self._starfile_data["particles"]

    @property
    @override
    def optics_data(self) -> pd.DataFrame:
        """The `pandas.DataFrame` of optics STAR file entries."""
        return self._starfile_data["optics"]

    @property
    def num_particles(self) -> int:
        """The number of particles in the STAR file."""
        return len(self._starfile_data["particles"])

    @property
    def num_optics_groups(self) -> int:
        """The number of optics groups in the STAR file."""
        return self._num_optics_groups

    @property
    def max_optics_groups(self) -> int:
        """The maximum number of optics groups allowed in
        the STAR file.

        !!! info

            Every time parameters are written via
            `parameter_file[...] = parameter_info` or
            `parameter_file.append(parameter_info)` a new optics
            group is created.

            Under the hood, the underlying STAR file
            data pre-allocates `max_optics_groups` entries for
            these writes and throws away any extra on calls to
            `parameter_file.save()`.
        """
        return len(self._starfile_data["optics"])

    @max_optics_groups.setter
    def max_optics_groups(self, value: int):
        if self.max_optics_groups < value:
            optics_data = self._starfile_data["optics"]
            self._starfile_data["optics"] = optics_data.reindex(index=range(value))
        elif self.max_optics_groups == value:
            pass
        else:
            raise ValueError(
                "Tried to set `parameter_file.max_optics_groups = ...`, "
                "but found that the current `max_optics_groups` "
                f"({self.max_optics_groups}) is greater than the new "
                f"value ({value}). Decreasing `max_optics_groups` is "
                "not supported."
            )

    @property
    @override
    def loads_metadata(self) -> bool:
        """Whether or not to load the following information:

        ```python
        parameter_info = parameter_file[index]
        assert "metadata" in parameter_info  # True
        ```

        The `metadata` is a `pandas.DataFrame` that includes the
        the STAR file rows *not* loaded by the `parameter_file`.
        For example, 'rlnClassNumber'.
        """
        return self._options["loads_metadata"]

    @loads_metadata.setter
    @override
    def loads_metadata(self, value: bool):
        self._options["loads_metadata"] = value

    @property
    @override
    def loads_envelope(self) -> bool:
        """Whether or not to load the following information:

        ```python
        parameter_info = parameter_file[index]
        assert parameter_info["transfer_theory"].envelope is not None  # True
        ```
        """
        return self._options["loads_envelope"]

    @loads_envelope.setter
    @override
    def loads_envelope(self, value: bool):
        self._options["loads_envelope"] = value

    @property
    def make_image_config(self) -> MakeImageConfig:
        """A function that returns a
        [`cryojax.simulator.BasicImageConfig`](https://michael-0brien.github.io/cryojax/api/simulator/config/)
        with signature `make_image_config(shape, pixel_size, voltage_in_kilovolts)`.
        """  # noqa: E501
        return self._options["make_image_config"]

    @make_image_config.setter
    def make_image_config(self, value: MakeImageConfig):
        self._options["make_image_config"] = value

    def _increment_optics_group(self):
        with self._lock:
            if self.num_optics_groups == self.max_optics_groups:
                raise IndexError(
                    "The number of optics groups in the STAR file exceeded "
                    "the number allocated in the "
                    "`parameter_file.starfile_data['optics']` dataframe. Try "
                    "increasing the value of `max_optics_groups` via "
                    "`parameter_file.max_optics_groups = ...`."
                )
            next_optics_group_index, next_optics_array_index = (
                self._next_optics_group_index,
                self._num_optics_groups,
            )
            self._next_optics_group_index += 1
            self._num_optics_groups += 1
        return next_optics_group_index, next_optics_array_index

    def __deepcopy__(self, memo):
        # This is a nasty hack to ensure that this class
        # works with its `parameter_file.copy()` routine!
        # The `threading.Lock()` class does not work under
        # `deepcopy`, so we need to manually reset it.
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k == "_lock":
                continue
            setattr(result, k, copy.deepcopy(v, memo))

        result._lock = threading.Lock()

        return result


class RelionParticleDataset(
    AbstractParticleDataset[_ParticleStackInfo, _ParticleStackLike]
):
    """A dataset that wraps a RELION particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html) format.
    """

    def __init__(
        self,
        parameter_file: AbstractRelionParticleParameterFile,
        path_to_relion_project: str | pathlib.Path,
        mode: Literal["r", "w"] = "r",
        *,
        mrcfile_options: dict[str, Any] = {},
        only_images: bool = False,
    ):
        """**Arguments:**

        - `path_to_relion_project`:
            In RELION STAR files, only a relative path is added to the
            'rlnImageName' column. This is relative to the path to the
            "project", which is given by this parameter.
        - `parameter_file`:
            The [`cryospax.RelionParticleParameterFile`][].
        - `mode`:
            - If `mode = 'w'`, the dataset is prepared to write new
            *images*. This is done by removing 'rlnImageName' from
            `parameter_file.starfile_data`, if it exists at all.
            does not have a column 'rlnImageName' and image files
            are not yet written.
            - If `mode = 'r'`, images are read from the 'rlnImageName'
            stored in the `parameter_file.starfile_data`.
        - `mrcfile_options`:
            A dictionary with options for MRC file writing. This accepts
            the following keys:
            - `'prefix'`:
                A `str` which acts as the prefix to the filenames. If this
                is equal to `'img'`, then the filename for image stack 0 will
                be called "img_00000.mrcs", for `delimiter = '_'` and
                `n_characters = 5`. By default, `'img'`.
            - `'output_folder'`:
                A `str` or `pathlib.Path` type where to write MRC files,
                relative to the `path_to_relion_project`.
            - `'n_characters'`:
                An `int` for the number of characters to write the filename
                number string. If this is equal to `5`, then the filename
                for image stack 0 will be called "img_00000.mrcs", for
                `delimiter = '_'` and `prefix = 'img'`. By default, `5`.
             - `'delimiter'`:
                A `str` for the delimiter between the filename prefix
                and number string. If this is equal to `'_'`, then the
                filename for image stack 0 will be called "img_00000.mrcs",
                for `n_characters = 5` and `prefix = 'img'`. By default, `'img'`.
            - `'overwrite'`:
                If `True`, overwrite existing MRC file path if it exists. By
        - `only_images`:
            If `False`, load parameters and images. Otherwise, load only images.
        """
        # Set properties. First, core properties of the dataset, starting
        # with the `mode``
        self._mode = _validate_mode(mode)
        # ... then, the `parameter_file`. If `mode = 'w'` but
        # the images already exist, we should make a copy in case
        # those images are being used elsewhere
        self._parameter_file = parameter_file
        # ... properties common to reading and writing images
        self._path_to_relion_project = pathlib.Path(path_to_relion_project)
        # ... properties for reading images
        self._only_images = only_images
        # ... properties for writing images
        self._mrcfile_options = _dict_to_mrcfile_options(mrcfile_options)
        # Now, initialize for `mode = 'r'` vs `mode = 'w'`
        images_exist = "rlnImageName" in parameter_file.particle_data.columns
        project_exists = self.path_to_relion_project.exists()
        if mode == "w":
            # Keep track of the file index
            self._next_file_index = 0
            # Write empty "rlnImageName" column (defaults to NaN values)
            self.particle_data["rlnImageName"] = pd.Series(dtype=str)
            # Make the project directory, if it does not yet exist
            if not project_exists:
                self.path_to_relion_project.mkdir(parents=True, exist_ok=False)
        else:
            if not images_exist:
                raise OSError(
                    "Could not find column 'rlnImageName' in the STAR file. "
                    "When using `mode = 'r'`, the STAR file must have this "
                    "column. To write images in a STAR file, "
                    "set `mode = 'w'`."
                )
            if not project_exists:
                raise FileNotFoundError(
                    "`RelionParticleDataset` opened in "
                    "'mode = `r`', but the RELION project directory "
                    "`path_to_relion_project` does not exist. "
                    "To write images in a STAR file for a new RELION project, "
                    "set `mode = 'w'`."
                )
            maximum_file_index = _get_maximum_file_index(
                self.particle_data["rlnImageName"]
            )
            if pd.isna(maximum_file_index):
                raise OSError(
                    "Tried to retrieve the maximum file index in the "
                    "STAR file 'rlnImageName' column, but did not correctly "
                    "parse the filenames. Make sure that filenames end "
                    "with some kind of numerical indexing, e.g. "
                    "'img_00000.mrcs'."
                )
            self._next_file_index = maximum_file_index + 1
        self._lock = threading.Lock()

    @classmethod
    def empty(
        cls: type[Self],
        path_to_starfile: str | pathlib.Path,
        path_to_relion_project: str | pathlib.Path,
        *,
        max_optics_groups: int = 1,
        exist_ok: bool = False,
        num_particles: int = 0,
        mrcfile_options: dict[str, Any] = {},
    ) -> Self:
        """Convenience wrapper for intializing a new
        [`cryospax.RelionParticleDataset`][] in `mode = 'w'`
        and writing via `dataset[index] = particle_info` or
        `dataset.append(particle_info)`.

        See [`cryospax.RelionParticleParameterFile.__init__`][]
        and [`cryospax.RelionParticleDataset.__init__`][]
        for more information.
        """
        parameter_file = RelionParticleParameterFile.empty(
            path_to_starfile,
            exist_ok=exist_ok,
            num_particles=num_particles,
            max_optics_groups=max_optics_groups,
        )
        return cls(
            parameter_file,
            path_to_relion_project,
            mode="w",
            mrcfile_options=mrcfile_options,
        )

    @classmethod
    def load(
        cls: type[Self],
        path_to_starfile: str | pathlib.Path,
        path_to_relion_project: str | pathlib.Path,
        *,
        selection_filter: dict[str, Callable] = {},
        # For writing via `dataset[index] = value`
        max_optics_groups: int | None = None,
        mrcfile_options: dict[str, Any] = {},
        # For loading via `value = dataset[index]`
        loads_metadata: bool = False,
        loads_envelope: bool = False,
        make_image_config: MakeImageConfig = default_make_image_config,
        only_images: bool = False,
    ) -> Self:
        """Convenience wrapper for loading a
        [`cryospax.RelionParticleDataset`][] in `mode = 'r'`
        and reading via `particle_info = dataset[index]` or
        writing via `dataset[index] = particle_info`.

        See [`cryospax.RelionParticleParameterFile.__init__`][]
        and [`cryospax.RelionParticleDataset.__init__`][]
        for more information.
        """
        parameter_file = RelionParticleParameterFile.load(
            path_to_starfile,
            selection_filter=selection_filter,
            loads_metadata=loads_metadata,
            loads_envelope=loads_envelope,
            make_image_config=make_image_config,
            max_optics_groups=max_optics_groups,
        )
        return cls(
            parameter_file,
            path_to_relion_project,
            mode="r",
            only_images=only_images,
            mrcfile_options=mrcfile_options,
        )

    @override
    def __getitem__(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " N"]
    ) -> _ParticleStackInfo:
        """Load dataset with `value = dataset[...]` syntax,
        where `value` is a dictionary with keys:

        - 'images':
            An image or image stack to write to an MRC file. This
            key is required.
        - 'parameters':
            See [`cryospax.RelionParticleParameterFile`][] for more
            information. This key is not included if
            `only_images = True`.
        """  # noqa: E501
        if not self.only_images:
            # Load images and parameters. First, read parameters
            # and metadata from the STAR file
            # ... read parameters
            parameters = self.parameter_file[index]
            # ... and the metadata
            if self.parameter_file.loads_metadata:
                particle_data_at_index = parameters["metadata"]
            else:
                particle_data_at_index = self.parameter_file.get_metadata(index)
            _validate_rln_image_name_exists(particle_data_at_index, index)
            # ... grab shape
            shape = parameters["image_config"].shape
            # ... load stack of images
            images = _load_image_stack_from_mrc(
                shape, particle_data_at_index, self.path_to_relion_project
            )
            # ... make sure images and parameters have same leading dim
            if parameters["pose"].offset_x_in_angstroms.ndim == 0:
                images = np.squeeze(images)

            return _ParticleStackInfo(parameters=parameters, images=images)
        else:
            # Otherwise, do not read parameters to more efficiently read images. First,
            # validate the dataset index.
            num_particles = len(self.parameter_file)
            _validate_dataset_index(type(self), index, num_particles)
            # ... read particle data at the requested indices
            particle_data_at_index = self.parameter_file.particle_data.iloc[index]
            if isinstance(particle_data_at_index, pd.Series):
                particle_data_at_index = particle_data_at_index.to_frame().T
            _validate_rln_image_name_exists(particle_data_at_index, index)
            # ... grab shape by reading the optics group
            optics_group = _get_optics_group_from_particle_data(
                particle_data_at_index, self.parameter_file.optics_data
            )
            image_size = int(optics_group["rlnImageSize"])
            shape = (image_size, image_size)
            # ... load stack of images
            images = _load_image_stack_from_mrc(
                shape, particle_data_at_index, self.path_to_relion_project
            )
            # ... make sure image leading dim matches with index query
            if isinstance(index, int) or (
                isinstance(index, np.ndarray) and index.size == 0
            ):
                images = np.squeeze(images)

            return _ParticleStackInfo(images=images)

    @override
    def __len__(self) -> int:
        """The number of particles in the STAR file."""
        return len(self.parameter_file)

    @override
    def __setitem__(
        self, index: int | slice | Int[np.ndarray, ""], value: _ParticleStackLike
    ):
        """Set dataset entries with `dataset[...] = value` syntax,
        where `value` is a dictionary with keys:

        - 'images':
            An image or image stack to write to an MRC file. This
            key is required.
        - 'parameters':
            See [`cryospax.RelionParticleParameterFile`][] for more
            information. This key is optional.
        """
        if isinstance(index, Int[np.ndarray, "_"]):  # type: ignore
            raise ValueError(
                "When setting `dataset[index] = ...`, "
                "it is not supported to pass `index` as a 1D numpy-array."
            )
        if not isinstance(value, dict):
            raise TypeError(
                "When setting `dataset[index] = foo`, "
                "foo must be a dictionary with key "
                "'images' and optionally key 'parameters'."
            )
        images, parameters = _unpack_particle_stack_dict(value)
        if parameters is not None:
            self.parameter_file[index] = parameters
        n_particles = len(self.parameter_file)
        index_array = np.atleast_1d(_index_to_array(index, n_particles))
        self.write_images(index_array, images, parameters=parameters)

    @override
    def append(self, value: _ParticleStackLike):
        """Add an entry or entries to the dataset with
        `dataset.append(value)` syntax, where `value`
        is a dictionary with keys:

        - 'images':
            An image or image stack to write to the MRC file.
        - 'parameters':
            See [`cryospax.RelionParticleParameterFile`][] for more
            information.

        Both keys are required.
        """
        if not isinstance(value, dict):
            raise TypeError(
                "When appending `dataset.append(foo)`, "
                "`foo` must be a dictionary with keys "
                "'images' and 'parameters'."
            )
        images, parameters = _unpack_particle_stack_dict(value)
        if parameters is None:
            raise ValueError(
                "When appending dictionary `foo` as `dataset.append(foo)`, "
                "`foo` must have key 'parameters'."
            )
        start = len(self.parameter_file)
        # Append parameters. This automatically sets the 'rlnImageName'
        # column to NaNs
        self.parameter_file.append(parameters)
        # Write images
        stop = len(self.parameter_file)
        index_array = np.arange(start, stop, dtype=int)
        self.write_images(index_array, images, parameters=parameters)

    @override
    def write_images(
        self,
        index_array: Int[np.ndarray, " _"],
        images: Float[NDArrayLike, "... _ _"],
        parameters: _ParticleParameterLike | None = None,
    ):
        """Write images to the dataset given `images` and `parameters`.

        This function is wrapped by [`cryospax.RelionParticleDataset.append`][]
        and [`cryospax.RelionParticleDataset.__setitem__`][] and in most cases
        these APIs are preferred.
        """
        # Get relevant metadata
        parameter_file = self.parameter_file
        if parameters is None:
            optics_group = _get_optics_group_from_particle_data(
                parameter_file.particle_data.iloc[index_array], parameter_file.optics_data
            )
            pixel_size, dim = (
                float(optics_group["rlnImagePixelSize"]),
                int(optics_group["rlnImageSize"]),
            )
        else:
            pixel_size, dim = (
                float(np.atleast_1d(parameters["image_config"].pixel_size)[0]),
                parameters["image_config"].shape[0],
            )
        if not (images.ndim in [2, 3] and images.shape[-2:] == (dim, dim)):
            raise ValueError(
                "Image(s) must be of "
                "shape `(n_images, dim, dim)` or `(dim, dim)`. "
                f"Tried writing image(s) of "
                f"shape {images.shape}."
            )
        # Prepare to write images
        if images.ndim == 2:
            images = images[None, ...]
        n_images, _ = images.shape[0], images.shape[1]

        # Convert index into 1D ascending numpy array
        n_indices = index_array.size
        if n_images != n_indices:
            raise ValueError(
                "Tried to set dataset elements with an inconsistent number "
                f"of images. Found that the number of images was {n_images}, "
                f"while the number of dataset indices was {n_indices}."
            )
        # Get absolute path to the filename, as well as the 'rlnImageName'
        # column
        file_index = self._increment_file_index()
        path_to_filename, rln_image_names = _make_rln_image_name(
            index_array,
            file_index,
            self.mrcfile_options,
            self.path_to_relion_project,
        )
        # Set the STAR file column
        parameter_file.particle_data.loc[
            parameter_file.particle_data.index[index_array], "rlnImageName"
        ] = rln_image_names
        # ... and write the images to disk
        try:
            write_image_stack_to_mrc(
                images.astype(jnp.float32),
                pixel_size,
                path_to_filename,
                overwrite=self.mrcfile_options["overwrite"],
                compression=self.mrcfile_options["compression"],
            )
        except Exception as err:
            raise OSError(
                "Error occurred when writing image stack to MRC "
                "file. Most likely, the filename the writer "
                f"chose ({str(path_to_filename)}) already "
                "exists. Try changing the "
                "`RelionParticleDataset.mrcfile_options`. "
                f"The error message was:\n{err}"
            )

    @property
    @override
    def parameter_file(self) -> AbstractRelionParticleParameterFile:
        """The `parameter_file` that reads/writes from the STAR
        file. This is modified internally during calls to
        `dataset[...] = value`.

        This cannot be modified after initialization.
        """
        return self._parameter_file

    @property
    def path_to_relion_project(self) -> pathlib.Path:
        """The path to the RELION project. Paths in the
        STAR file are relative to this directory.

        This cannot be modified after initialization.
        """
        return self._path_to_relion_project

    @property
    @override
    def mode(self) -> Literal["r", "w"]:
        """Whether or not the `dataset` was
        instantiated in reading ('r') or writing ('w') mode.

        This cannot be modified after initialization.
        """
        return self._mode  # type: ignore

    @property
    def particle_data(self) -> pd.DataFrame:
        """The `pandas.DataFrame` of particle STAR file entries.

        Alias to [`cryospax.AbstractRelionParticleParameterFile.particle_data`][].
        """
        return self.parameter_file.particle_data

    @property
    def optics_data(self) -> pd.DataFrame:
        """The `pandas.DataFrame` of optics STAR file entries.

        Alias to [`cryospax.AbstractRelionParticleParameterFile.optics_data`][].
        """
        return self.parameter_file.optics_data

    @property
    def mrcfile_options(self) -> _MrcfileOptions:
        """Settings for writing MRC files with. See
        [`cryospax.RelionParticleDataset.__init__`][]
        for more information.
        """
        return self._mrcfile_options

    @mrcfile_options.setter
    def mrcfile_options(self, value: dict[str, Any]):
        self._mrcfile_options = _dict_to_mrcfile_options(value)

    @property
    def only_images(self) -> bool:
        """If `True`, load images and *not* parameters. This gives
        better performance when it is not necessary to load parameters.

        ```python
        dataset.only_images = True
        particle_info = dataset[0]
        assert "images" in particle_info  # True
        assert "parameters" not in particle_info  # True
        ```
        """
        return self._only_images

    @only_images.setter
    def only_images(self, value: bool):
        self._only_images = value

    def __deepcopy__(self, memo):
        # This is a nasty hack to ensure that this class
        # works with its `dataset.copy()` routine!
        # The `threading.Lock()` class does not work under
        # `deepcopy`, so we need to manually reset it.
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k == "_lock":
                continue
            setattr(result, k, copy.deepcopy(v, memo))

        result._lock = threading.Lock()

        return result

    def _increment_file_index(self):
        with self._lock:
            file_index = self._next_file_index
            self._next_file_index += 1
        return file_index


def _load_starfile_data(
    path_to_starfile: pathlib.Path,
    mode: Literal["r", "w"],
    selection_filter: dict[str, Callable],
    exist_ok: bool,
    num_particles: int,
    max_optics_groups: int | None,
) -> tuple[_StarfileData, tuple[int, int]]:
    if mode == "r":
        if path_to_starfile.exists():
            starfile_data = read_starfile(path_to_starfile)
            _validate_starfile_data(starfile_data)
            # Handle particle entries
            if len(selection_filter) > 0:
                starfile_data = _select_particles(starfile_data, selection_filter)
            # Handle optics group entries
            optics_data = starfile_data["optics"]
            num_optics_groups, max_optics_group_index = (
                len(optics_data),
                optics_data["rlnOpticsGroup"].max().item(),
            )
            if pd.isna(max_optics_group_index):
                raise OSError(
                    "Tried to parse the optics group 'rlnOpticsGroup' column to "
                    "retrieve its maximum value, but found that "
                    "it had a NaN value. Make sure that your STAR file is correctly "
                    "formatted."
                )
            if max_optics_groups is None:
                max_optics_groups = 2 * num_optics_groups
            starfile_data["optics"] = optics_data.reindex(index=range(max_optics_groups))
        else:
            raise FileNotFoundError(
                f"Set `mode = '{mode}'`, but STAR file {str(path_to_starfile)} does not "
                "exist. To write a new STAR file, set `mode = 'w'`."
            )
    else:
        if path_to_starfile.exists() and not exist_ok:
            raise FileExistsError(
                f"Set `mode = 'w'`, but STAR file {str(path_to_starfile)} already "
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
                            for column, dtype in RELION_DEFAULT_PARTICLE_ENTRIES
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


def _validate_mode(mode: str) -> Literal["r", "w"]:
    if mode not in ["r", "w"]:
        raise ValueError(
            f"Passed unsupported `mode = {mode}`. Supported modes are 'r' and 'w'."
        )
    return mode  # type: ignore


def _select_particles(
    starfile_data: dict[str, pd.DataFrame], selection_filter: dict[str, Callable]
) -> dict[str, pd.DataFrame]:
    particle_data = starfile_data["particles"]
    boolean_mask = pd.Series(True, index=particle_data.index)
    for key in selection_filter:
        if key in particle_data.columns:
            fn = selection_filter[key]
            column = particle_data[key]
            base_error_message = (
                f"Error filtering key '{key}' in the `selection_filter`. "
                f"To filter the STAR file entries, `selection_filter['{key}']`"
                "must be a function that takes in an array and returns a "
                "boolean mask."
            )
            if isinstance(selection_filter[key], Callable):
                try:
                    mask_at_column = fn(column)
                except Exception as err:
                    raise ValueError(
                        f"{base_error_message} "
                        "When calling the function, caught an error:\n"
                        f"{err}"
                    )
                if not pd.api.types.is_bool_dtype(mask_at_column):
                    raise ValueError(
                        f"{base_error_message} "
                        "Found that the function did not return "
                        "a boolean dtype."
                    )
            else:
                raise ValueError(base_error_message)
            # Update mask
            boolean_mask = mask_at_column & boolean_mask
        else:
            raise ValueError(
                f"Included key '{key}' in the `selection_filter`, "
                "but this entry could not be found in the STAR file. "
                "The `selection_filter` must be a dictionary whose "
                "keys are strings in the STAR file and whose values "
                "are functions that take in columns and return boolean "
                "masks."
            )
    # Select particles using mask
    starfile_data["particles"] = particle_data[boolean_mask]

    return starfile_data


#
# STAR file reading
#
def _make_pytrees_from_starfile(
    particle_data,
    optics_data,
    loads_envelope,
    make_image_config,
) -> tuple[BasicImageConfig, ContrastTransferTheory, EulerAnglePose]:
    float_dtype = jax.dtypes.canonicalize_dtype(float)
    # Load CTF parameters. First from particle data
    defocus_in_angstroms = (
        np.asarray(particle_data["rlnDefocusU"], dtype=float_dtype)
        + np.asarray(particle_data["rlnDefocusV"], dtype=float_dtype)
    ) / 2
    astigmatism_in_angstroms = np.asarray(
        particle_data["rlnDefocusU"], dtype=float_dtype
    ) - np.asarray(particle_data["rlnDefocusV"], dtype=float_dtype)
    astigmatism_angle = np.asarray(particle_data["rlnDefocusAngle"], dtype=float_dtype)
    phase_shift = np.asarray(particle_data["rlnPhaseShift"], dtype=float_dtype)
    # Then from optics data
    batch_shape = (
        () if defocus_in_angstroms.ndim == 0 else (defocus_in_angstroms.shape[0],)
    )
    spherical_aberration_in_mm = np.full(
        batch_shape, np.asarray(optics_data["rlnSphericalAberration"], dtype=float_dtype)
    )
    amplitude_contrast_ratio = np.full(
        batch_shape, np.asarray(optics_data["rlnAmplitudeContrast"], dtype=float_dtype)
    )
    ctf_params = (
        defocus_in_angstroms,
        astigmatism_in_angstroms,
        astigmatism_angle,
        spherical_aberration_in_mm,
        amplitude_contrast_ratio,
        phase_shift,
    )
    # Envelope parameters
    if loads_envelope:
        b_factor, scale_factor = (
            (
                np.asarray(particle_data["rlnCtfBfactor"], dtype=float_dtype)
                if "rlnCtfBfactor" in particle_data.keys()
                else None
            ),
            (
                np.asarray(particle_data["rlnCtfScalefactor"], dtype=float_dtype)
                if "rlnCtfScalefactor" in particle_data.keys()
                else None
            ),
        )
    else:
        b_factor, scale_factor = None, None
    # Image config parameters
    pixel_size = np.asarray(optics_data["rlnImagePixelSize"], dtype=float_dtype)
    voltage_in_kilovolts = np.asarray(optics_data["rlnVoltage"], dtype=float_dtype)
    # Pose parameters. Values for the pose are optional,
    # so look to see if each key is present
    particle_keys = particle_data.keys()
    # Read the pose. first, xy offsets
    rln_origin_x_angst = (
        particle_data["rlnOriginXAngst"] if "rlnOriginXAngst" in particle_keys else 0.0
    )
    rln_origin_y_angst = (
        particle_data["rlnOriginYAngst"] if "rlnOriginYAngst" in particle_keys else 0.0
    )
    # ... rot angle
    rln_angle_rot = (
        particle_data["rlnAngleRot"] if "rlnAngleRot" in particle_keys else 0.0
    )
    # ... tilt angle
    if "rlnAngleTilt" in particle_keys:
        rln_angle_tilt = particle_data["rlnAngleTilt"]
    elif "rlnAngleTiltPrior" in particle_keys:  # support for helices
        rln_angle_tilt = particle_data["rlnAngleTiltPrior"]
    else:
        rln_angle_tilt = 0.0
    # ... psi angle
    if "rlnAnglePsi" in particle_keys:
        # Relion uses -999.0 as a placeholder for an un-estimated in-plane
        # rotation
        if isinstance(particle_data["rlnAnglePsi"], pd.Series):
            # ... check if all values are equal to -999.0. If so, just
            # replace the whole pandas.Series with 0.0
            if (
                particle_data["rlnAnglePsi"].nunique() == 1
                and particle_data["rlnAnglePsi"].iloc[0] == -999.0
            ):
                rln_angle_psi = 0.0
            # ... otherwise, replace -999.0 values with 0.0
            else:
                rln_angle_psi = particle_data["rlnAnglePsi"].where(
                    lambda x: x != -999.0, 0.0
                )
        else:
            # ... if the column is just equal to a float, then
            # directly check if it is equal to -999.0
            rln_angle_psi = (
                0.0
                if particle_data["rlnAnglePsi"] == -999.0
                else particle_data["rlnAnglePsi"]
            )
    elif "rlnAnglePsiPrior" in particle_keys:  # support for helices
        rln_angle_psi = particle_data["rlnAnglePsiPrior"]
    else:
        rln_angle_psi = 0.0
    # Now, flip the sign of the translations and transpose rotations.
    # RELION's convention thinks about the translation as "undoing" the translation
    # and rotation in the image
    maybe_make_full = lambda param: (
        np.full(batch_shape, param)
        if len(batch_shape) > 0 and param.shape == ()
        else param
    )
    pose_params = tuple(
        maybe_make_full(x)
        for x in (
            -np.asarray(rln_origin_x_angst, dtype=float_dtype),
            -np.asarray(rln_origin_y_angst, dtype=float_dtype),
            -np.asarray(rln_angle_rot, dtype=float_dtype),
            -np.asarray(rln_angle_tilt, dtype=float_dtype),
            -np.asarray(rln_angle_psi, dtype=float_dtype),
        )
    )
    # Now, create cryojax objects. Do this on the CPU
    cpu_device = jax.devices(backend="cpu")[0]
    with jax.default_device(cpu_device):
        # First, create the `BasicImageConfig`
        image_size = int(optics_data["rlnImageSize"])
        image_shape = (image_size, image_size)
        image_config = filter_device_get(
            make_image_config(image_shape, pixel_size, voltage_in_kilovolts)
        )
        # ... now the `ContrastTransferTheory`
        envelope = (
            _make_envelope_function(scale_factor, b_factor) if loads_envelope else None
        )
        transfer_theory_params = (*ctf_params, envelope)
        transfer_theory = _make_transfer_theory(*transfer_theory_params)  # type: ignore
        # ... finally the `EulerAnglePose`
        pose = _make_pose(*pose_params)
    # Now, convert arrays to numpy in case the user wishes to do preprocessing
    pytree_dynamic, pytree_static = eqx.partition(
        (image_config, transfer_theory, pose), eqx.is_array
    )
    pytree_dynamic = jax.tree.map(lambda x: np.asarray(x), pytree_dynamic)
    image_config, transfer_theory, pose = eqx.combine(pytree_dynamic, pytree_static)

    return image_config, transfer_theory, pose


def _make_pose(offset_x, offset_y, phi, theta, psi):
    _make_fn = lambda _x, _y, _phi, _theta, _psi: EulerAnglePose(
        _x, _y, _phi, _theta, _psi
    )
    if offset_x.ndim == 1:
        _make_fn = eqx.filter_vmap(_make_fn)
    return _make_fn(offset_x, offset_y, phi, theta, psi)


def _make_envelope_function(amp, b_factor):
    if b_factor is None and amp is None:
        warnings.warn(
            "`loads_envelope` was set to True, but no envelope parameters were found. "
            "Setting envelope as None. "
            "Make sure your starfile is correctly formatted or set "
            "`loads_envelope=False`."
        )
        return None

    elif b_factor is None and amp is not None:
        return eqx.tree_at(lambda x: x.value, FourierConstant(1.0), amp)
    else:
        if amp is None:
            amp = np.asarray(1.0) if b_factor.ndim == 0 else np.ones_like(b_factor)
        return eqx.tree_at(
            lambda x: (x.amplitude, x.b_factor),
            FourierGaussian(1.0, 1.0),
            (amp, b_factor),
        )


def _make_transfer_theory(defocus, astig, angle, sph, ac, ps, env=None):
    ctf = eqx.tree_at(
        lambda x: (
            x.defocus_in_angstroms,
            x.astigmatism_in_angstroms,
            x.astigmatism_angle,
            x.spherical_aberration_in_mm,
        ),
        AstigmaticCTF(),
        (defocus, astig, angle, sph),
    )
    transfer_theory = ContrastTransferTheory(
        ctf, envelope=env, amplitude_contrast_ratio=0.1, phase_shift=0.0
    )

    return eqx.tree_at(
        lambda x: (x.amplitude_contrast_ratio, x.phase_shift), transfer_theory, (ac, ps)
    )


def _load_image_stack_from_mrc(
    shape: tuple[int, int],
    particle_dataframe_at_index: pd.DataFrame,
    path_to_relion_project: str | pathlib.Path,
) -> Float[np.ndarray, "... y_dim x_dim"]:
    # Load particle image stack rlnImageName
    rln_image_names = particle_dataframe_at_index["rlnImageName"].reset_index(drop=True)
    if rln_image_names.convert_dtypes().dtype != "string":
        raise TypeError(
            "The 'rlnImageName' column was expected to be type string. "
            f"Instead, found type {rln_image_names.dtype}."
        )
    # Split the pandas.Series into two: one for the image index
    # and another for the filename
    split = rln_image_names.str.split("@").str
    # ... relion convention starts indexing at 1, not 0
    mrc_index, filenames = split[0], split[1]
    if filenames.isnull().any() or mrc_index.isnull().any():
        raise ValueError(
            "Could not parse filenames from the 'rlnImageName' "
            "column. Check to make sure these are the correct format, "
            "for example '00000@path/to/image-00000.mrcs'."
        )
    mrc_index = mrc_index.astype(int) - 1
    # Allocate memory for stack
    n_images = len(filenames)
    image_stack = np.empty((n_images, *shape), dtype=float)
    # Loop over filenames to fill stack
    unique_filenames = filenames.unique()
    for filename in unique_filenames:
        # Get the MRC indices
        mrc_index_at_filename = mrc_index[filename == filenames]
        particle_index_at_filename = mrc_index_at_filename.index
        path_to_filename = pathlib.Path(path_to_relion_project, filename)
        with mrcfile.mmap(path_to_filename, mode="r", permissive=True) as mrc:
            mrc_data = np.asarray(mrc.data)
            mrc_ndim = mrc_data.ndim
            mrc_shape = mrc_data.shape if mrc_ndim == 2 else mrc_data.shape[1:]

            if shape != mrc_shape:
                raise ValueError(
                    f"The shape of the MRC with filename {filename} "
                    "was found to not have the same shape loaded from "
                    "the 'rlnImageSize'. Check your MRC files and also "
                    "the STAR file optics group formatting."
                )
            image_stack[particle_index_at_filename] = (
                mrc_data if mrc_ndim == 2 else mrc_data[mrc_index_at_filename]
            )

    return image_stack


def _validate_dataset_index(cls, index, n_rows):
    index_error_msg = lambda idx: (
        f"The index at which the `{cls.__name__}` was accessed was out of bounds! "
        f"The number of rows in the dataset is {n_rows}, but you tried to "
        f"access the index {idx}."
    )
    # ... pandas has bad error messages for its indexing
    if isinstance(index, (int, np.integer)):  # type: ignore
        if index > n_rows - 1:
            raise IndexError(index_error_msg(index))
    elif isinstance(index, slice):
        if index.start is not None and index.start > n_rows - 1:
            raise IndexError(index_error_msg(index.start))
    elif isinstance(index, np.ndarray):
        if index.ndim > 1:
            raise IndexError(
                f"Tried to index {cls.__name__} by a numpy "
                f"array, but found that the array had `ndim = {index.ndim}`. "
                "Only 0-d and 1-d numpy arrays are supported."
            )
    else:
        raise IndexError(
            f"Indexing with the type {type(index)} is not supported by "
            f"`{cls.__name__}`. Indexing by integers is supported, one-dimensional "
            "fancy indexing is supported, and numpy-array indexing is supported. "
            "For example, like `value = dataset[0]`, "
            "`value = dataset[0:5]`, "
            "or `value = dataset[np.array([1, 4, 3, 2])]`."
        )


def _validate_starfile_data(starfile_data: dict[str, pd.DataFrame]):
    if "particles" not in starfile_data.keys():
        raise ValueError("Missing key 'particles' in `starfile.read` output.")
    else:
        required_particle_keys, _ = zip(*RELION_REQUIRED_PARTICLE_ENTRIES)
        if not set(required_particle_keys).issubset(
            set(starfile_data["particles"].keys())
        ):
            raise ValueError(
                "Missing required keys in starfile 'particles' group. "
                f"Required keys are {required_particle_keys}."
            )
    if "optics" not in starfile_data.keys():
        raise ValueError("Missing key 'optics' in `starfile.read` output.")
    else:
        required_optics_keys, _ = zip(*RELION_REQUIRED_OPTICS_ENTRIES)
        if not set(required_optics_keys).issubset(set(starfile_data["optics"].keys())):
            raise ValueError(
                "Missing required keys in starfile 'optics' group. "
                f"Required keys are {required_optics_keys}."
            )


def _validate_rln_image_name_exists(particle_data, index):
    if "rlnImageName" not in particle_data.columns:
        raise OSError(
            "Tried to read STAR file for "
            f"`RelionParticleDataset` index = {index}, "
            "but no entry found for 'rlnImageName'."
        )


#
# Working with pytrees for I/O
#
def _unpack_particle_stack_dict(
    value: _ParticleStackLike,
) -> tuple[Float[NDArrayLike, "... y_dim x_dim"], _ParticleParameterLike | None]:
    if "images" in value:
        images = value["images"]
    else:
        raise ValueError(
            "When passing dictionary `foo` as `dataset.append(foo)` or "
            "`dataset[index] = foo`, `foo` must have key `images`."
        )
    if "parameters" in value:
        parameters = value["parameters"]
    else:
        parameters = None

    return images, parameters


#
# STAR file writing. First, functions for writing parameters
#
def _validate_parameters(parameters: _ParticleParameterLike, force_keys: bool = False):
    if force_keys:
        if not {"image_config", "transfer_theory", "pose"}.issubset(parameters):
            raise ValueError(
                "When passing dictionary `foo` as `parameter_file.append(foo)` "
                "`foo` must have keys 'pose', 'transfer_theory', and 'image_config'."
            )
    if "image_config" in parameters:
        if not isinstance(parameters["image_config"], BasicImageConfig):
            raise TypeError(
                "Found that dict key 'image_config' was "
                "not type `cryojax.simulator.BasicImageConfig`. "
                f"Instead, it was type "
                f"{type(parameters['image_config']).__name__}."
            )
    if "transfer_theory" in parameters:
        if not isinstance(parameters["transfer_theory"], ContrastTransferTheory):
            raise TypeError(
                "Found that dict key 'transfer_theory' was "
                "not type `cryojax.simulator.ContrastTransferTheory`. "
                f"Instead, it was type "
                f"{type(parameters['transfer_theory']).__name__}."
            )
    if "pose" in parameters:
        if not isinstance(parameters["pose"], EulerAnglePose):
            raise TypeError(
                "Found that dict key 'pose' was "
                "not type `cryojax.simulator.EulerAnglePose`. "
                f"Instead, it was type "
                f"{type(parameters['pose']).__name__}."
            )


def _parameters_to_optics_data(
    parameters: _ParticleParameterLike, optics_group_index: int
) -> pd.DataFrame:
    if {"image_config", "transfer_theory"}.issubset(parameters):
        shape = parameters["image_config"].shape
        if shape[0] == shape[1]:
            dim = shape[0]
        else:
            raise ValueError(
                "When adding optics group to STAR file, found "
                "non-square shape in `image_config.shape`. Only "
                "square shapes are supported."
            )
        pixel_size = parameters["image_config"].pixel_size
        voltage_in_kilovolts = parameters["image_config"].voltage_in_kilovolts
        amplitude_contrast_ratio = parameters["transfer_theory"].amplitude_contrast_ratio
        if isinstance(parameters["transfer_theory"].ctf, AstigmaticCTF):
            spherical_aberration_in_mm = getattr(
                parameters["transfer_theory"].ctf, "spherical_aberration_in_mm"
            )
        else:
            raise ValueError(
                "When adding optics group or particle to STAR file, "
                "`transfer_theory.ctf` must be type "
                "`AstigmaticCTF`. Instead, got type "
                f"{type(parameters['transfer_theory'].ctf).__name__}."
            )
        optics_group_dict = {
            "rlnOpticsGroup": optics_group_index,
            "rlnImageSize": dim,
            "rlnImagePixelSize": pixel_size,
            "rlnVoltage": voltage_in_kilovolts,
            "rlnSphericalAberration": spherical_aberration_in_mm,
            "rlnAmplitudeContrast": amplitude_contrast_ratio,
        }
        for k, v in optics_group_dict.items():
            if isinstance(v, Array | np.ndarray):
                arr = np.atleast_1d(np.asarray(v))
                if arr.size > 1:
                    if np.unique(arr).size > 1:
                        raise ValueError(
                            "Tried to fill a RELION optics group entry with an array "
                            "that has multiple unique values. Optics group compatible "
                            "arrays such as `BasicImageConfig.pixel_size` "
                            "must be either scalars or arrays all with the same value. "
                            f"Error occurred when filling '{k}' with array {v}."
                        )
                optics_group_dict[k] = arr.ravel()[0, None]
            else:
                optics_group_dict[k] = [v]
        optics_data = pd.DataFrame.from_dict(optics_group_dict)

        return optics_data
    else:
        raise ValueError(
            "Tried to add optics group to the STAR file, but "
            "parameter dictionary did not include 'image_config' "
            "and 'transfer_theory' keys. If you are setting parameters "
            "`parameter_file[index] = dict(pose=pose)` make sure "
            "`parameter_file.updates_optics_group = False`."
        )


def _parameters_to_particle_data(
    parameters: _ParticleParameterLike, optics_group_index: int | None
) -> pd.DataFrame:
    particles_dict = {}
    if "pose" in parameters:
        # Now, pose parameters
        pose = parameters["pose"]
        if not isinstance(pose, EulerAnglePose):
            raise ValueError(
                "When adding particle to STAR file, "
                "`pose` must be type "
                "`EulerAnglePose`. Instead, got type "
                f"{type(pose).__name__}."
            )
        if pose.offset_in_angstroms.ndim == 2:
            particles_dict["rlnOriginXAngst"] = -pose.offset_in_angstroms[:, 0]
            particles_dict["rlnOriginYAngst"] = -pose.offset_in_angstroms[:, 1]
        elif pose.offset_in_angstroms.ndim == 1:
            particles_dict["rlnOriginXAngst"] = -pose.offset_in_angstroms[0]
            particles_dict["rlnOriginYAngst"] = -pose.offset_in_angstroms[1]
        else:
            raise RuntimeError(
                "Internal `cryojax` error when loading translations to STAR file."
            )
        particles_dict["rlnAngleRot"] = -pose.phi_angle
        particles_dict["rlnAngleTilt"] = -pose.theta_angle
        particles_dict["rlnAnglePsi"] = -pose.psi_angle
        # Now, broadcast parameters to same dimension
        n_particles = pose.offset_x_in_angstroms.size
        for k, v in particles_dict.items():
            if v.shape == ():
                particles_dict[k] = np.full((n_particles,), np.asarray(v))
            elif v.size == n_particles:
                particles_dict[k] = v.ravel()
            else:
                raise ValueError(
                    "Found inconsistent number of particles "
                    "when adding particle to STAR file. "
                    "When running `parameter_file[index] = foo` "
                    "or `parameter_file.append(foo)`, make sure `foo` "
                    "has arrays that are scalars or all have the same "
                    "number of dimensions."
                )
    else:
        raise ValueError(
            "Tried to modify parameters in STAR file, but "
            "parameter dictionary did not include 'pose' "
            "key. If you are setting parameters "
            "`parameter_file[index] = foo` make sure "
            "`foo` is, for example, `foo = dict(pose=EulerAnglePose(...))`."
        )
    # Fill CTF parameters
    if "transfer_theory" in parameters:
        transfer_theory = parameters["transfer_theory"]
        if isinstance(transfer_theory.ctf, AstigmaticCTF):
            if pose.offset_z_in_angstroms is None:
                defocus_offset = 0.0
            else:
                defocus_offset = pose.offset_z_in_angstroms
            particles_dict["rlnDefocusU"] = (
                transfer_theory.ctf.defocus_in_angstroms
                + defocus_offset
                + transfer_theory.ctf.astigmatism_in_angstroms / 2
            )
            particles_dict["rlnDefocusV"] = (
                transfer_theory.ctf.defocus_in_angstroms
                + defocus_offset
                - transfer_theory.ctf.astigmatism_in_angstroms / 2
            )
            particles_dict["rlnDefocusAngle"] = transfer_theory.ctf.astigmatism_angle
        else:
            raise ValueError(
                "When adding particle to STAR file, "
                "`transfer_theory.ctf` must be type "
                "`AstigmaticCTF`. Instead, got type "
                f"{type(parameters['transfer_theory'].ctf).__name__}."
            )

        if isinstance(transfer_theory.envelope, FourierGaussian):
            particles_dict["rlnCtfBfactor"] = transfer_theory.envelope.b_factor
            particles_dict["rlnCtfScalefactor"] = transfer_theory.envelope.amplitude
        elif isinstance(transfer_theory.envelope, FourierConstant):
            particles_dict["rlnCtfScalefactor"] = transfer_theory.envelope.value
        elif transfer_theory.envelope is None:
            pass
        else:
            raise ValueError(
                "When adding particle to STAR file, "
                "`transfer_theory.envelope` must be "
                "type `cryojax.ndimage.FourierGaussian` "
                "or `cryojax.ndimage.FourierConstant`, or `None`. Got "
                f"{type(transfer_theory.envelope).__name__}."
            )
        particles_dict["rlnPhaseShift"] = transfer_theory.phase_shift
    if optics_group_index is not None:
        # Now, miscellaneous parameters
        particles_dict["rlnOpticsGroup"] = np.full(
            (n_particles,), optics_group_index, dtype=int
        )
    # Make sure parameters are numpy arrays
    particles_dict = jax.tree.map(
        lambda x: np.asarray(x) if isinstance(x, Array) else x, particles_dict
    )
    particle_data = pd.DataFrame.from_dict(particles_dict)
    # Finally, see if the particle parameters has metadata and if so,
    # add this
    if "metadata" in parameters:
        metadata = parameters["metadata"]
        if n_particles != metadata.index.size:
            raise ValueError(
                "When adding custom metadata to STAR file "
                "with `parameter_file[index] = foo` or `parameter_file.append(foo)`, "
                "found the number of particles "
                "in `foo['metadata']` was inconsistent with the "
                "number of particles in `foo['pose']`."
            )
        # Add metadata to dataframe
        particle_data = pd.concat(
            [particle_data, metadata], axis="columns", verify_integrity=True
        )
    return particle_data


def _parse_optics_group_index(particle_data_at_index: pd.DataFrame | pd.Series) -> int:
    # ... read optics data
    optics_group_indices = np.unique(
        np.atleast_1d(np.asarray(particle_data_at_index["rlnOpticsGroup"]))
    )
    if optics_group_indices.size > 1:
        warnings.warn(
            "Tried to read multiple particles at once that belong "
            "to different optics groups, but this is not yet "
            "implemented. Proceeding by using the first optics group "
            "encountered...",
            category=UserWarning,
            stacklevel=2,
        )
    optics_group_index = optics_group_indices[0]

    return int(optics_group_index)


def _get_optics_group_from_index(
    optics_data: pd.DataFrame, optics_group_index: int
) -> pd.Series:
    return optics_data[optics_data["rlnOpticsGroup"] == optics_group_index].iloc[0]


def _get_optics_group_from_particle_data(
    particle_data_at_index: pd.DataFrame | pd.Series, optics_data: pd.DataFrame
) -> pd.Series:
    optics_group_index = _parse_optics_group_index(particle_data_at_index)
    return _get_optics_group_from_index(optics_data, optics_group_index)


#
# Now, functions for writing image files
#
def _dict_to_mrcfile_options(d: dict[str, Any]) -> _MrcfileOptions:
    _options_keys = {
        "prefix",
        "output_folder",
        "delimiter",
        "n_characters",
        "overwrite",
        "compression",
    }
    if not set(d.keys()).issubset(_options_keys):
        raise ValueError(
            "Expected that dictionary `mrcfile_options` passed to "
            "`RelionParticleDataset(..., mrcfile_options=...)` "
            f"had a subset of keys {_options_keys}, but found that it "
            f"had keys {set(d.keys())}."
        )
    prefix = d["prefix"] if "prefix" in d else "img"
    output_folder = d["output_folder"] if "output_folder" in d else ""
    delimiter = d["delimiter"] if "delimiter" in d else "_"
    n_characters = d["n_characters"] if "n_characters" in d else 5
    overwrite = d["overwrite"] if "overwrite" in d else False
    compression = d["compression"] if "compression" in d else None
    return _MrcfileOptions(
        prefix=prefix,
        output_folder=output_folder,
        delimiter=delimiter,
        n_characters=n_characters,
        overwrite=overwrite,
        compression=compression,
    )


def _dict_to_options(d: dict[str, Any]) -> _Options:
    _options_keys = {
        "loads_metadata",
        "loads_envelope",
        "make_image_config",
    }
    if not set(d.keys()).issubset(_options_keys):
        raise ValueError(
            "Expected that dictionary `options` passed to "
            "`RelionParticleParameterFile(..., options=...)` "
            f"had a subset of keys {_options_keys}, but found that it "
            f"had keys {set(d.keys())}."
        )
    loads_metadata = d["loads_metadata"] if "loads_metadata" in d else False
    loads_envelope = d["loads_envelope"] if "loads_envelope" in d else False
    make_image_config = (
        d["make_image_config"] if "make_image_config" in d else _default_make_image_config
    )
    return _Options(
        loads_metadata=loads_metadata,
        loads_envelope=loads_envelope,
        make_image_config=make_image_config,
    )


def _index_to_array(indices: slice | int | np.ndarray, size: int) -> np.ndarray:
    if isinstance(indices, slice):
        return np.asarray(range(*indices.indices(size)))
    else:
        return np.asarray(indices, dtype=int)


def _make_rln_image_name(
    index: Int[np.ndarray, " _"],
    file_number: int,
    mrcfile_options: _MrcfileOptions,
    path_to_relion_project: pathlib.Path,
) -> tuple[pathlib.Path, list[str]]:
    prefix = mrcfile_options["prefix"]
    output_folder = mrcfile_options["output_folder"]
    delimiter = mrcfile_options["delimiter"]
    n_characters = mrcfile_options["n_characters"]
    # Generate filename
    file_number_fmt = _format_number_for_filename(file_number, n_characters=n_characters)
    relative_path_to_filename = str(
        pathlib.Path(output_folder, prefix + delimiter + file_number_fmt + ".mrcs")
    )
    # Finally, generate the 'rln_image_name' column, which includes the particle index
    rln_image_names = [
        _format_number_for_filename(int(i + 1), n_characters)
        + "@"
        + relative_path_to_filename
        for i in range(index.size)
    ]
    # Finally, the path to the filename
    path_to_filename = pathlib.Path(path_to_relion_project, relative_path_to_filename)

    return path_to_filename, rln_image_names


def _get_maximum_file_index(rln_image_name) -> int:
    # TODO: this function ~may~ silently fail if DataFrame is
    # incorrectly formatted!
    filenames = rln_image_name.str.split("@").str[-1]
    file_index = filenames.str.extract(r"(\d+)\.[^.]+$", expand=False)
    return file_index.astype("Int64").max(skipna=True).item()


def _format_number_for_filename(file_number: int, n_characters: int = 6):
    if file_number == 0:
        return "0" * n_characters
    else:
        n_digits = int(np.log10(file_number)) + 1
        return "0" * (n_characters - n_digits) + str(file_number)
