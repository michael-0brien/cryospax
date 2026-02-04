"""cryoJAX compatibility with [CSPARC](https://relion.readthedocs.io/en/release-5.0/)."""

import abc
import pathlib
import typing
from collections.abc import Callable
from typing import Any, Literal, TypedDict, cast
from typing_extensions import NotRequired, override

import equinox as eqx
import jax
import mrcfile
import numpy as np
import pandas as pd
from cryojax.jax_util import NDArrayLike
from cryojax.simulator import (
    AxisAnglePose,
    BasicImageConfig,
    ContrastTransferTheory,
)
from jaxtyping import Float, Int

from .._io import read_csparc_data
from .._misc import filter_device_get
from .base_dataset import (
    AbstractParticleDataset,
    AbstractParticleParameterFile,
)
from .common import (
    MakeImageConfig,
    _dict_to_options,
    _make_envelope_function,
    _make_transfer_theory,
    _validate_dataset_index,
    _validate_mode,
)


# CSPARC column entries
CSPARC_INSTRUMENT_ENTRIES = [
    ("blob/shape", "Int64"),
    ("ctf/accel_kv", "Float64"),
    ("blob/psize_A", "Float64"),
]
CSPARC_CTF_ENTRIES = [
    ("ctf/amp_contrast", "Float64"),
    ("ctf/cs_mm", "Float64"),
    ("ctf/df1_A", "Float64"),
    ("ctf/df2_A", "Float64"),
    ("ctf/df_angle_rad", "Float64"),
    ("ctf/phase_shift_rad", "Float64"),
]
CSPARC_POSE_ENTRIES = [
    ("alignments3D/pose", "Float64"),
    ("alignments3D/shift", "Float64"),
    ("alignments_class_0/pose", "Float64"),
    ("alignments_class_0/shift", "Float64"),
]


# Required entries for loading
CSPARC_REQUIRED_PARTICLE_ENTRIES = [
    *CSPARC_CTF_ENTRIES,
]
CSPARC_SUPPORTED_PARTICLE_ENTRIES = [
    *CSPARC_REQUIRED_PARTICLE_ENTRIES,
    *CSPARC_POSE_ENTRIES,
    ("ctf/bfactor", "Float64"),
    ("ctf/scale", "Float64"),
]


if hasattr(typing, "GENERATING_DOCUMENTATION"):
    _ParticleParameterInfo = dict[str, Any]  # pyright: ignore[reportAssignmentType]
    _ParticleStackInfo = dict[str, Any]  # pyright: ignore[reportAssignmentType]
    _ParticleParameterLike = dict[str, Any]  # pyright: ignore[reportAssignmentType]
    _ParticleStackLike = dict[str, Any]  # pyright: ignore[reportAssignmentType]
    _Options = dict[str, Any]  # pyright: ignore[reportAssignmentType]
    _MrcfileSettings = dict[str, Any]  # pyright: ignore[reportAssignmentType]

else:
    from .common import _MrcfileSettings

    class _ParticleParameterInfo(TypedDict):
        """Parameters for a particle stack from RELION."""

        image_config: BasicImageConfig
        pose: AxisAnglePose
        transfer_theory: ContrastTransferTheory

        metadata: NotRequired[pd.DataFrame]

    class _ParticleStackInfo(TypedDict):
        """Particle stack info from RELION."""

        images: Float[np.ndarray, "... y_dim x_dim"]
        parameters: NotRequired[_ParticleParameterInfo]

    _ParticleParameterLike = dict[str, Any] | _ParticleParameterInfo
    _ParticleStackLike = dict[str, Any] | _ParticleStackInfo


class AbstractParticleCryoSparcFile(
    AbstractParticleParameterFile[_ParticleParameterInfo, _ParticleParameterLike]
):
    @property
    @override
    def path_to_output(self) -> pathlib.Path:
        return self.path_to_csfile

    @path_to_output.setter
    @override
    def path_to_output(self, value: str | pathlib.Path):
        self.path_to_csfile = value

    @property
    @abc.abstractmethod
    def path_to_csfile(self) -> pathlib.Path:
        raise NotImplementedError

    @path_to_csfile.setter
    @abc.abstractmethod
    def path_to_csfile(self, value: str | pathlib.Path):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def csfile_data(self) -> pd.DataFrame:
        raise NotImplementedError

    @csfile_data.setter
    @abc.abstractmethod
    def csfile_data(self, value: dict[str, pd.DataFrame]):
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
    def updates_optics_group(self) -> bool:
        raise NotImplementedError

    @updates_optics_group.setter
    @abc.abstractmethod
    def updates_optics_group(self, value: bool):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def make_image_config(self) -> MakeImageConfig:
        raise NotImplementedError

    @make_image_config.setter
    @abc.abstractmethod
    def make_image_config(self, value: MakeImageConfig):
        raise NotImplementedError


class CryoSparcParticleParameterFile(AbstractParticleCryoSparcFile):
    """A dataset that wraps a CSPARC particle stack in
    [CryoSPARC  ](https://guide.cryosparc.com/setup-configuration-and-management/software-system-guides/manipulating-.cs-files-created-by-cryosparc)
    format.


    **Example:**
    ```python
    from cryojax.dataset import CryoSparcParticleParameterFile
    from cryojax.io import read_csparc_data

    # For knowing how to set a filter it is useful to see how we
    # we read the cryosparc metadata

    csfile = read_csparc_data(
        "path/to/cryosparc/particles.cs"
    )
    print(csfile.head())

    # For example, to select only particles from class 0 from a 3D classification job:

    csparc_parameter_file = CryoSparcParticleParameterFile(
        path_to_csfile="path/to/cryosparc/particles.cs",
        selection_filter={
            "alignments3D/class": lambda x: x == 0
        },
    )


    ```
    """

    def __init__(
        self,
        path_to_csfile: str | pathlib.Path,
        mode: Literal["r", "w"] = "r",
        *,
        exists_ok: bool = False,
        selection_filter: dict[str, Callable] | None = None,
        options: dict[str, Any] = {},
    ):
        """**Arguments:**

         - `path_to_csfile`:
             The path to the CryoSPARC parameters file (`.cs`). If the path does not exist
             and `mode = 'w'`, an empty dataset will be created.
         - `mode`:
             - If `mode = 'r'`, the CryoSPARC parameters file at `path_to_csfile` is read
             into `CryoSparcParticleParameterFile.csfile_data`.
             - mode = 'w'` is not currently supported.
         - `exist_ok`:
             Currently not used for this type of dataset,
             as it is only relevant in writing mode.
        - `selection_filter`:
             A dictionary used to include only particular dataset elements.
             The keys of this dictionary should be any data entry in the CryoSPARC
             parameters file, while the values should be a function that takes in a
             column and returns a boolean mask for the column. For example,
             filter by class using
             `selection_filter["alignments3D/class"] = lambda x: x == 0`.
         - `options`:
             A dictionary of options for modifying the behavior of loading/writing.
             - 'loads_metadata':
                 If `True`, the resulting dict loads
                 the raw metadata from the CryoSPARC parameters file that is not otherwise
                 included into a `pandas.DataFrame`.
                 If this is set to `True`, note that dictionaries cannot pass through
                 JIT boundaries without removing the metadata.
                 By default, `False`.
             - 'loads_envelope':
                 If `True`, read in the parameters of the CTF envelope function, i.e.
                 "ctf/scale" and "ctf/bfactor".
                 By default, `False`.
             - 'updates_optics_group':
                 Currently not used for this type of dataset,
                 as it is only relevant in writing mode.
             - 'make_image_config':
                 A function with signature
                 `fn(shape, pixel_size, voltage_in_kilovolts)` that
                 returns a [`cryojax.simulator.BasicImageConfig`](https://michael-0brien.github.io/cryojax/api/simulator/config/)
                 class. Use this argument when it is desired to customize the `image_config`
                 returned from this class, i.e.
                 `value = parameter_file[0:7]; print(value["image_config"])`.
        """  # noqa: E501

        # Private attributes
        _mode = _validate_mode(mode)
        assert _mode == "r", (
            "Writing mode is not currently supported for CryoSPARC files."
        )
        self._options = _dict_to_options(options)
        self._mode = _mode

        # The CryoSPARC file data
        self._path_to_csfile = pathlib.Path(path_to_csfile)
        self._csfile_data = _load_csfile_data(self._path_to_csfile, selection_filter)

    @override
    def __getitem__(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " _"]
    ) -> _ParticleParameterInfo:
        """Load CryoSPARC file entries with `value = parameter_file[...]` syntax,
        where `value` is a dictionary with keys:

        - 'pose':
            The [`cryojax.simulator.AxisAnglePose`](https://michael-0brien.github.io/cryojax/api/simulator/pose/#cryojax.simulator.AxisAnglePose)
        - 'image_config':
            The [`cryojax.simulator.BasicImageConfig`](https://michael-0brien.github.io/cryojax/api/simulator/config/#cryojax.simulator.BasicImageConfig)
        - 'transfer_theory':
            The [`cryojax.simulator.ContrastTransferTheory`](https://michael-0brien.github.io/cryojax/api/simulator/transfer_theory/#cryojax.simulator.ContrastTransferTheory)
        - 'metadata':
            If `loads_metadata = True`, a `pandas.DataFrame` of entries
            *not* used when loading the `pose`, `image_config`, and
            `transfer_theory` (e.g. the 'alignments3D/class'). Otherwise, this
            key is not included.
        """  # noqa: E501
        # Validate index
        n_rows = self.csfile_data.shape[0]
        _validate_dataset_index(type(self), index, n_rows)
        # ... read particle data at the requested indice
        csparc_data_at_index = self.csfile_data.iloc[index]

        # Load the image stack and CryoSPARC file parameters
        image_config, transfer_theory, pose = _make_pytrees_from_csfile(
            csparc_data_at_index, self.loads_envelope, self.make_image_config
        )
        parameter_info = _ParticleParameterInfo(
            image_config=image_config, pose=pose, transfer_theory=transfer_theory
        )
        if self.loads_metadata:
            # ... convert to dataframe for serialization
            if isinstance(csparc_data_at_index, pd.Series):
                csparc_data_at_index = csparc_data_at_index.to_frame().T
            # ... no overlapping keys with loaded pytrees
            redundant_entry_labels, _ = list(zip(*CSPARC_SUPPORTED_PARTICLE_ENTRIES))
            columns = csparc_data_at_index.columns
            remove_columns = [
                column for column in columns if column in redundant_entry_labels
            ]
            metadata = csparc_data_at_index.drop(remove_columns, axis="columns")
            parameter_info["metadata"] = metadata

        return parameter_info

    @override
    def __len__(self) -> int:
        return len(self.csfile_data)

    @property
    @override
    def path_to_csfile(self) -> pathlib.Path:
        return self._path_to_medata

    @path_to_csfile.setter
    @override
    def path_to_csfile(self, value: str | pathlib.Path):
        self._path_to_medata = pathlib.Path(value)

    @property
    @override
    def csfile_data(self) -> pd.DataFrame:
        return self._csfile_data

    @csfile_data.setter
    @override
    def csfile_data(self, value: dict[str, pd.DataFrame]):
        raise NotImplementedError("csfile_data cannot be modified")

    @property
    def mode(self) -> Literal["r", "w"]:
        return self._mode  # type: ignore

    @property
    @override
    def loads_metadata(self) -> bool:
        return self._loads_metadata

    @loads_metadata.setter
    @override
    def loads_metadata(self, value: bool):
        self._loads_metadata = value

    @property
    @override
    def loads_envelope(self) -> bool:
        return self._loads_envelope

    @loads_envelope.setter
    @override
    def loads_envelope(self, value: bool):
        self._loads_envelope = value

    @property
    @override
    def updates_optics_group(self) -> bool:
        raise NotImplementedError

    @updates_optics_group.setter
    @override
    def updates_optics_group(self, value: bool):
        raise NotImplementedError

    @property
    def inverts_rotation(self) -> bool:
        return self._inverts_rotation

    @inverts_rotation.setter
    def inverts_rotation(self, value: bool):
        self._inverts_rotation = value

    @override
    def __setitem__(
        self,
        index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " _"],
        value: _ParticleParameterLike,
    ):
        raise NotImplementedError(
            "CryoSparcParticleParameterFile does not have a __setitem__ method"
        )

    @override
    def append(self, value: _ParticleParameterLike):
        raise NotImplementedError(
            "append is not supported for CryoSparcParticleParameterFile"
        )

    @override
    def save(
        self,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ):
        raise NotImplementedError(
            "saving is not supported for CryoSparcParticleParameterFile"
        )


class CryoSparcParticleDataset(
    AbstractParticleDataset[_ParticleStackInfo, _ParticleStackLike]
):
    """A dataset that wraps a CryoSPARC particle stack in
    [CryoSPARC](https://guide.cryosparc.com/setup-configuration-and-management/software-system-guides/manipulating-.cs-files-created-by-cryosparc)
    format.
    """

    def __init__(
        self,
        parameter_file: AbstractParticleCryoSparcFile,
        path_to_relion_project: str | pathlib.Path,
        mode: Literal["r", "w"] = "r",
        *,
        mrcfile_settings: dict[str, Any] = {},
        just_images: bool = False,
    ):
        """**Arguments:**

        - `path_to_relion_project`:
            In CryoSPARC files, only a relative path is added to the
            'blob/path' column. This is relative to the path to the
            "project", which is given by this parameter.
        - `parameter_file`:
            The `CryoSparcParticleParameterFile`.
         - `mode`:
             - If `mode = 'r'`, the CryoSPARC parameters file at `path_to_csfile` is read
             into `CryoSparcParticleParameterFile.csfile_data`.
             - mode = 'w'` is not currently supported.
        - mrcfile_settings:
            Currently not used for this type of dataset
            as writing mode is not implemented.
        - `just_images`:
            If `False`, load parameters and images. Otherwise, load only images.

        """
        # Set properties. First, core properties of the dataset, starting
        # with the `mode``
        _mode = _validate_mode(mode)
        assert _mode == "r", (
            "Writing mode is not currently supported for CryoSPARC datasets."
        )
        self._mode = _mode

        particle_data = parameter_file.csfile_data

        self._parameter_file = parameter_file
        # ... properties common to reading and writing images
        self._path_to_relion_project = pathlib.Path(path_to_relion_project)
        # ... properties for reading images
        self._just_images = just_images
        # ... properties for writing images
        self._mrcfile_settings = mrcfile_settings
        # Now, initialize for `mode = 'r'` vs `mode = 'w'`
        images_exist = "blob/path" in particle_data.columns
        project_exists = self.path_to_relion_project.exists()
        if not images_exist:
            raise OSError(
                "Could not find column 'blob/path' in the CryoSparc metadata file. "
            )
        if not project_exists:
            raise FileNotFoundError("The CSPARC project directory does not exist.")

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
            See [`cryospax.CryoSparcParticleParameterFile`][] for more
            information. This key is not included if
            `just_images = True`.
        """  # noqa: E501
        if not self.just_images:
            # Load images and parameters. First, read parameters
            # and metadata from the .cs file
            loads_metadata = self.parameter_file.loads_metadata
            self.parameter_file.loads_metadata = True
            # ... read parameters
            parameters = self.parameter_file[index]
            # ... validate the metadata
            csparc_data_at_index = cast(pd.DataFrame, parameters["metadata"])
            _validate_csparc_image_name_exists(csparc_data_at_index, index)
            # ... reset boolean to original value
            self.parameter_file.loads_metadata = loads_metadata
            if not loads_metadata:
                del parameters["metadata"]
            # ... grab shape
            shape = parameters["image_config"].shape
            # ... load stack of images
            images = _load_image_stack_from_mrc(
                shape, csparc_data_at_index, self.path_to_relion_project
            )
            # ... make sure images and parameters have same leading dim
            if parameters["pose"].offset_x_in_angstroms.ndim == 0:
                images = np.squeeze(images)

            return _ParticleStackInfo(parameters=parameters, images=images)

        else:
            # Otherwise, do not read parameters to more efficiently read images. First,
            # validate the dataset index.
            n_rows = self.parameter_file.csfile_data.shape[0]
            _validate_dataset_index(type(self), index, n_rows)
            # ... read particle data at the requested indices
            particle_data = self.parameter_file.csfile_data
            csparc_data_at_index = particle_data.iloc[index]
            if isinstance(csparc_data_at_index, pd.Series):
                csparc_data_at_index = csparc_data_at_index.to_frame().T
            _validate_csparc_image_name_exists(csparc_data_at_index, index)
            # ... grab shape by reading the optics group
            shape = tuple(int(x) for x in csparc_data_at_index["blob/shape"][0])
            shape = cast(tuple[int, int], shape)
            # ... load stack of images
            images = _load_image_stack_from_mrc(
                shape, csparc_data_at_index, self.path_to_relion_project
            )
            # ... make sure image leading dim matches with index query
            if isinstance(index, int) or (
                isinstance(index, np.ndarray) and index.size == 0
            ):
                images = np.squeeze(images)

            return _ParticleStackInfo(images=images)

    @override
    def __setitem__(
        self, index: int | slice | Int[np.ndarray, ""], value: _ParticleStackLike
    ):
        raise NotImplementedError(
            "CryoSparcParticleDataset does not have a __setitem__ method"
        )

    @override
    def append(self, value: _ParticleStackLike):
        raise NotImplementedError("append is not supported for CryoSparcParticleDataset")

    @override
    def write_images(
        self,
        index_array: Int[np.ndarray, " _"],
        images: Float[NDArrayLike, "... _ _"],
        parameters: _ParticleParameterLike | None = None,
    ):
        raise NotImplementedError(
            "writing images is not supported for CryoSparcParticleDataset"
        )

    @override
    def __len__(self) -> int:
        return len(self.parameter_file)

    @property
    @override
    def parameter_file(self) -> AbstractParticleCryoSparcFile:
        return self._parameter_file

    @property
    @override
    def mode(self) -> Literal["r", "w"]:
        """Whether or not the `dataset` was
        instantiated in reading ('r') or writing ('w') mode.

        This cannot be modified after initialization.
        """
        return self._mode  # type: ignore

    @property
    def path_to_relion_project(self) -> pathlib.Path:
        """The path to the RELION project. Paths in the
        CryoSPARC file are relative to this directory.

        This cannot be modified after initialization.
        """
        return self._path_to_relion_project

    @property
    def mrcfile_settings(self) -> _MrcfileSettings:
        """Settings for writing MRC files with. See
        [`cryospax.CryoSparcParticleDataset.__init__`][]
        for more information.
        """
        return self._mrcfile_settings

    @mrcfile_settings.setter
    def mrcfile_settings(self, value: dict[str, Any]):
        raise NotImplementedError(
            "mrcfile_settings cannot be modified for CryoSparcParticleDataset."
        )

    @property
    def just_images(self) -> bool:
        """If `True`, load images and *not* parameters. This gives
        better performance when it is not necessary to load parameters.

        ```python
        dataset.just_images = True
        particle_info = dataset[0]
        assert "images" in particle_info  # True
        assert "parameters" not in particle_info  # True
        ```
        """
        return self._just_images

    @just_images.setter
    def just_images(self, value: bool):
        self._just_images = value


def _load_csfile_data(
    path_to_csfile: pathlib.Path,
    selection_filter: dict[str, Callable] | None,
) -> pd.DataFrame:
    if path_to_csfile.exists():
        csfile_data = read_csparc_data(path_to_csfile)
        _validate_csfile_data(csfile_data)
        if selection_filter is not None:
            csfile_data = _select_particles(csfile_data, selection_filter)
    else:
        raise FileNotFoundError(
            f"CryoSparc parameters file {str(path_to_csfile)} does not exist."
        )

    return csfile_data


def _select_particles(
    csfile_data: pd.DataFrame, selection_filter: dict[str, Callable]
) -> pd.DataFrame:
    boolean_mask = pd.Series(True, index=csfile_data.index)
    for key in selection_filter:
        if key in csfile_data.columns:
            fn = selection_filter[key]
            column = csfile_data[key]
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
                "but this entry could not be found in the CryoSparc metadata file. "
                "The `selection_filter` must be a dictionary whose "
                "keys are strings in the STAR file and whose values "
                "are functions that take in columns and return boolean "
                "masks."
            )
    # Select particles using mask
    csfile_data = csfile_data[boolean_mask]

    return csfile_data


#
# CryoSPARC file reading
#
def _make_pytrees_from_csfile(
    csparc_data,
    loads_envelope,
    make_image_config,
) -> tuple[BasicImageConfig, ContrastTransferTheory, AxisAnglePose]:
    float_dtype = jax.dtypes.canonicalize_dtype(float)
    # Load CTF parameters. First from particle data
    defocus_in_angstroms = (
        np.asarray(csparc_data["ctf/df1_A"], dtype=float_dtype)
        + np.asarray(csparc_data["ctf/df2_A"], dtype=float_dtype)
    ) / 2
    astigmatism_in_angstroms = np.asarray(
        csparc_data["ctf/df1_A"], dtype=float_dtype
    ) - np.asarray(csparc_data["ctf/df2_A"], dtype=float_dtype)
    astigmatism_angle = np.rad2deg(
        np.asarray(csparc_data["ctf/df_angle_rad"], dtype=float_dtype)
    )
    phase_shift = np.rad2deg(
        np.asarray(csparc_data["ctf/phase_shift_rad"], dtype=float_dtype)
    )
    # Then from optics data
    batch_shape = (
        () if defocus_in_angstroms.ndim == 0 else (defocus_in_angstroms.shape[0],)
    )
    spherical_aberration_in_mm = np.asarray(csparc_data["ctf/cs_mm"], dtype=float_dtype)
    amplitude_contrast_ratio = np.asarray(
        csparc_data["ctf/amp_contrast"], dtype=float_dtype
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
                np.asarray(csparc_data["ctf/bfactor"], dtype=float_dtype)
                if "ctf/bfactor" in csparc_data.keys()
                else None
            ),
            (
                np.asarray(csparc_data["ctf/scale"], dtype=float_dtype)
                if "ctf/scale" in csparc_data.keys()
                else None
            ),
        )
    else:
        b_factor, scale_factor = None, None
    # Image config parameters
    pixel_size = np.asarray(csparc_data["blob/psize_A"], dtype=float_dtype)
    voltage_in_kilovolts = np.asarray(csparc_data["ctf/accel_kv"], dtype=float_dtype)
    if len(batch_shape) > 0:
        pixel_size = pixel_size[0]
        voltage_in_kilovolts = voltage_in_kilovolts[0]
    # Pose parameters. Values for the pose are optional,
    # so look to see if each key is present
    particle_keys = csparc_data.keys()
    # Read the pose. first, xy offsets

    if "alignments3D/shift" in particle_keys:
        csparc_pose_shift = np.array([s for s in csparc_data["alignments3D/shift"]])
    elif "alignments_class_0/shift" in particle_keys:
        csparc_pose_shift = np.array([s for s in csparc_data["alignments_class_0/shift"]])
    else:
        csparc_pose_shift = np.array([0.0, 0.0])

    if "alignments3D/pose" in particle_keys:
        csparc_pose_angles = np.array(
            [angles for angles in csparc_data["alignments3D/pose"]]
        )
    elif "alignments_class_0/pose" in particle_keys:
        csparc_pose_angles = np.array(
            [angles for angles in csparc_data["alignments_class_0/pose"]]
        )

    else:
        csparc_pose_angles = np.array([0.0, 0.0, 0.0])

    csparc_pose_angles = np.rad2deg(csparc_pose_angles)
    # Now transform the angles and shift to the AxisAnglePose convention
    if len(batch_shape) > 0:
        pose_shift = csparc_pose_shift * pixel_size[:, None]
    else:
        pose_shift = csparc_pose_shift * pixel_size

    # Now, flip the sign of the translations and transpose rotations.
    maybe_make_full = lambda param: (
        np.full(batch_shape, param)
        if len(batch_shape) > 0 and param.shape == ()
        else param
    )
    # AxisAngle inversion is not as simple as negating the angles, so
    # we do not do it here.
    pose_params = (
        -maybe_make_full(pose_shift),
        maybe_make_full(csparc_pose_angles),  # don't invert here, do it later
    )

    # Now, create cryojax objects. Do this on the CPU
    cpu_device = jax.devices(backend="cpu")[0]
    with jax.default_device(cpu_device):
        # First, create the `BasicImageConfig`
        if len(batch_shape) > 0:
            image_shape = tuple(int(x) for x in csparc_data["blob/shape"].values[0])
        else:
            image_shape = tuple(int(x) for x in csparc_data["blob/shape"])
        image_config = filter_device_get(
            make_image_config(image_shape, pixel_size, voltage_in_kilovolts)
        )

        # ... now the `ContrastTransferTheory`
        envelope = (
            _make_envelope_function(scale_factor, b_factor) if loads_envelope else None
        )
        transfer_theory_params = (*ctf_params, envelope)
        transfer_theory = _make_transfer_theory(*transfer_theory_params)  # type: ignore

        # ... finally the `AxisAnglePose`
        pose = _invert_rotation(_make_pose(*pose_params))

    # Now, convert arrays to numpy in case the user wishes to do preprocessing
    pytree_dynamic, pytree_static = eqx.partition(
        (image_config, transfer_theory, pose), eqx.is_array
    )
    pytree_dynamic = jax.tree.map(lambda x: np.asarray(x), pytree_dynamic)
    image_config, transfer_theory, pose = eqx.combine(pytree_dynamic, pytree_static)

    return image_config, transfer_theory, pose


def _make_pose(shift, euler_vector):
    _make_fn = lambda _shift, _euler_vector: AxisAnglePose(
        _shift[0],
        _shift[1],
        _euler_vector,
    )
    if shift.ndim == 2:
        _make_fn = eqx.filter_vmap(_make_fn)
    return _make_fn(shift, euler_vector)


def _invert_rotation(pose: AxisAnglePose) -> AxisAnglePose:
    if pose.offset_in_angstroms.ndim == 1:
        return pose.to_inverse_rotation()
    else:
        return eqx.filter_vmap(lambda p: p.to_inverse_rotation())(pose)


def _load_image_stack_from_mrc(
    shape: tuple[int, int],
    particle_dataframe_at_index: pd.DataFrame,
    path_to_relion_project: str | pathlib.Path,
) -> Float[np.ndarray, "... y_dim x_dim"]:
    # Load particle image stack rlnImageName
    mrc_filenames_and_indices = (
        particle_dataframe_at_index[["blob/path", "blob/idx"]]
        .copy()
        .reset_index(drop=True)
    )
    mrc_filenames_and_indices["idx_in_df"] = mrc_filenames_and_indices.index.copy()
    try:
        mrc_filenames_and_indices.loc[:, "blob/path"] = mrc_filenames_and_indices[
            "blob/path"
        ].astype(str)
    except ValueError as err:
        raise TypeError(
            "The 'blob/path' entry in the CryoSparc metadata could not be converted"
            f"to string. Caught error:\n{err}"
        )

    # groupby filename to get indices
    grouped_filenames = mrc_filenames_and_indices.groupby("blob/path").agg(list)

    # Allocate memory for stack
    n_images = len(mrc_filenames_and_indices)
    image_stack = np.empty((n_images, *shape), dtype=float)
    # Loop over filenames to fill stack
    for filename in grouped_filenames.index:
        # Get the MRC indices
        path_to_filename = pathlib.Path(path_to_relion_project, filename)
        with mrcfile.mmap(path_to_filename, mode="r", permissive=True) as mrc:
            mrc_data = np.asarray(mrc.data)
            mrc_ndim = mrc_data.ndim
            mrc_shape = mrc_data.shape if mrc_ndim == 2 else mrc_data.shape[1:]

            if shape != mrc_shape:
                raise ValueError(
                    f"The shape of the MRC with filename {filename} "
                    "was found to not have the same shape loaded from "
                    "the 'blob/shape'. Check your MRC files and also "
                    "the .cs file formatting."
                )
            idx_in_filename = np.array(
                grouped_filenames.loc[filename, "blob/idx"], dtype=int
            )
            idx_in_df = np.array(grouped_filenames.loc[filename, "idx_in_df"], dtype=int)
            image_stack[idx_in_df] = (
                mrc_data if mrc_ndim == 2 else mrc_data[idx_in_filename]
            )

    return image_stack


def _validate_csfile_data(csfile_data: pd.DataFrame):
    required_particle_keys, _ = zip(*CSPARC_REQUIRED_PARTICLE_ENTRIES)
    if not set(required_particle_keys).issubset(set(csfile_data.keys())):
        raise ValueError(
            "Missing required keys in .cs file items. "
            f"Required keys are {required_particle_keys}."
        )


def _validate_csparc_image_name_exists(particle_data, index):
    if "blob/path" not in particle_data.columns:
        raise OSError(
            "Tried to read CryoSparc metadata file for "
            f"`RelionParticleStackDataset` index = {index}, "
            "but no entry found for 'blob/path'."
        )
