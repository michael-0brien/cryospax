"""
Routines for .cs file serialization
"""

import pathlib
from typing import Any

import equinox as eqx
import numpy as np
import pandas as pd
from cryojax.rotations import SO3, convert_quaternion_to_euler_angles

from ._utils import _validate_filename


# The RELION columns that live in the 'optics' block, i.e. those that define an
# optics group, and those that live in the 'particles' block. These are the
# columns that `read_csparc_file_as_star` is able to convert from CryoSPARC
RELION_OPTICS_COLUMNS = [
    "rlnImagePixelSize",
    "rlnVoltage",
    "rlnSphericalAberration",
    "rlnAmplitudeContrast",
    "rlnImageSize",
]
RELION_PARTICLE_COLUMNS = [
    "rlnImageName",
    "rlnMicrographName",
    "rlnDefocusU",
    "rlnDefocusV",
    "rlnDefocusAngle",
    "rlnPhaseShift",
    "rlnCtfBfactor",
    "rlnOriginXAngst",
    "rlnOriginYAngst",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
]


def read_csparc_file(
    filename: str | pathlib.Path,
    passthrough_filename: str | pathlib.Path | None = None,
) -> pd.DataFrame:
    """Read a CryoSPARC `.cs` file into a `pandas.DataFrame` using `numpy`.

    No conversion is done here: the columns of the returned `pandas.DataFrame`
    are the CryoSPARC fields themselves, e.g. `'blob/psize_A'`. Byte strings are
    decoded to `str` and multi-dimensional fields, e.g. `'blob/shape'`, are stored
    as columns of `numpy` arrays.

    **Arguments:**

    - `filename`:
        The path where to read the CryoSPARC file. This must include
        a '.cs' extension.
    - `passthrough_filename`:
        An optional passthrough `.cs` file, whose fields are merged into the
        result on the particle `'uid'`. Fields already present in `filename`
        are taken from `filename`.

    **Returns:**

    A `pandas.DataFrame` with one row per particle and one column per
    CryoSPARC field.
    """
    _validate_filename(filename, mode="r", suffix="cs")
    csparc_data = _csparc_array_to_dataframe(np.load(filename))
    if passthrough_filename is not None:
        _validate_filename(passthrough_filename, mode="r", suffix="cs")
        passthrough_data = _csparc_array_to_dataframe(np.load(passthrough_filename))
        # Passthrough files repeat some of the fields of the main file. Keep the
        # main file's values so that the merge does not generate '_x'/'_y' columns
        duplicate_columns = [
            column
            for column in passthrough_data.columns
            if column != "uid" and column in csparc_data.columns
        ]
        csparc_data = csparc_data.merge(
            passthrough_data.drop(columns=duplicate_columns), on="uid", how="inner"
        )

    return csparc_data


def read_csparc_file_as_starfile(
    filename: str | pathlib.Path,
    passthrough_filename: str | pathlib.Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Read a CryoSPARC `.cs` file and convert it to RELION STAR file data.

    This reads the file with `read_csparc_file`, converts the CryoSPARC fields
    to their RELION equivalents, and splits the result into the two blocks of a
    RELION STAR file.

    CryoSPARC stores optics parameters per particle, while RELION stores them
    once per optics group. The RELION optics group is taken to be the CryoSPARC
    exposure group `'ctf/exp_group_id'`, and each exposure group becomes one row
    of the `'optics'` block, holding the optics parameters of its first particle.
    If the CryoSPARC file has no `'ctf/exp_group_id'` field, all particles are
    assigned to a single optics group.

    !!! info "Matching `pyem`"

        This follows [`pyem`](https://github.com/asarnow/pyem). In particular,
        exposure groups whose optics parameters happen to be equal are *not*
        merged into one optics group, and the CryoSPARC index is taken to be
        zero-based, so that `'rlnOpticsGroup'` is `'ctf/exp_group_id'` plus one.
        A one-based exposure group index is therefore numbered from two.

        Note that if a single exposure group contains particles with *differing*
        optics parameters, only those of its first particle are kept.

    **Arguments:**

    - `filename`:
        The path where to read the CryoSPARC file. This must include
        a '.cs' extension.
    - `passthrough_filename`:
        An optional passthrough `.cs` file. See `read_csparc_file`.

    **Returns:**

    A dictionary `dict(optics=..., particles=...)` of `pandas.DataFrame`s, in
    the format returned by `read_starfile`. The `'optics'` block has one row per
    optics group, and the `'particles'` block has one row per particle with an
    `'rlnOpticsGroup'` column pointing into the `'optics'` block.
    """
    csparc_data = read_csparc_file(filename, passthrough_filename)

    return _convert_csparc_data_to_starfile_data(csparc_data)


def _csparc_array_to_dataframe(csparc_array: np.ndarray) -> pd.DataFrame:
    names = csparc_array.dtype.names
    if names is None:
        raise OSError(
            "Tried to read a CryoSPARC file, but it did not contain a structured "
            "array of named fields."
        )
    columns = {}
    for name in names:
        values = csparc_array[name]
        # Handle byte strings
        if values.dtype.kind == "S":
            values = np.char.decode(values, "utf-8")
        # Handle multi-dimensional fields (stored as a column of arrays)
        columns[name] = list(values) if values.ndim > 1 else values

    return pd.DataFrame(columns)


def _convert_csparc_data_to_starfile_data(
    csparc_data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    relion_data = _convert_csparc_columns_to_relion(csparc_data)
    optics_data, particle_data = _split_optics_and_particle_data(relion_data)
    return dict(optics=optics_data, particles=particle_data)


def _convert_csparc_columns_to_relion(csparc_data: pd.DataFrame) -> pd.DataFrame:
    """Convert CryoSPARC fields to their RELION equivalents, as a single
    per-particle `pandas.DataFrame`.
    """
    columns = csparc_data.columns
    relion_data: dict[str, Any] = {}

    # --- Optics group ---
    # The CryoSPARC exposure group is the RELION optics group. Following `pyem`,
    # the CryoSPARC index is taken to be zero-based and is shifted to RELION's
    # one-based index
    if "ctf/exp_group_id" in columns:
        exposure_group = _column_to_array(csparc_data, "ctf/exp_group_id")
        optics_group = exposure_group.astype(np.int64) + 1
        # A RELION optics group must be one or greater. This is already the case
        # for a zero-based or one-based exposure group, and is only not the case
        # if CryoSPARC wrote a negative index, e.g. a '-1' placeholder
        if optics_group.size > 0 and optics_group.min() < 1:
            optics_group += 1 - optics_group.min()
        relion_data["rlnOpticsGroup"] = optics_group
    else:
        relion_data["rlnOpticsGroup"] = np.ones(len(csparc_data), dtype=np.int64)

    # --- Optics fields ---
    pixel_size = None
    if "blob/psize_A" in columns:
        pixel_size = _column_to_array(csparc_data, "blob/psize_A")
        relion_data["rlnImagePixelSize"] = pixel_size

    if "blob/shape" in columns:
        shape = _column_to_array(csparc_data, "blob/shape")
        if not (shape[:, 0] == shape[:, 1]).all():
            raise ValueError("Non-square images are not supported.")
        relion_data["rlnImageSize"] = shape[:, 0]

    if "ctf/accel_kv" in columns:
        relion_data["rlnVoltage"] = _column_to_array(csparc_data, "ctf/accel_kv")
    if "ctf/cs_mm" in columns:
        relion_data["rlnSphericalAberration"] = _column_to_array(csparc_data, "ctf/cs_mm")
    if "ctf/amp_contrast" in columns:
        relion_data["rlnAmplitudeContrast"] = _column_to_array(
            csparc_data, "ctf/amp_contrast"
        )

    # --- Particle fields ---
    # CTF
    if "ctf/defocus_u" in columns:
        relion_data["rlnDefocusU"] = _column_to_array(csparc_data, "ctf/defocus_u")
    elif "ctf/df1_A" in columns:
        relion_data["rlnDefocusU"] = _column_to_array(csparc_data, "ctf/df1_A")

    if "ctf/defocus_v" in columns:
        relion_data["rlnDefocusV"] = _column_to_array(csparc_data, "ctf/defocus_v")
    elif "ctf/df2_A" in columns:
        relion_data["rlnDefocusV"] = _column_to_array(csparc_data, "ctf/df2_A")

    if "ctf/defocus_angle" in columns:
        relion_data["rlnDefocusAngle"] = np.degrees(
            _column_to_array(csparc_data, "ctf/defocus_angle")
        )
    elif "ctf/df_angle_rad" in columns:
        relion_data["rlnDefocusAngle"] = np.degrees(
            _column_to_array(csparc_data, "ctf/df_angle_rad")
        )

    if "ctf/phase_shift_rad" in columns:
        relion_data["rlnPhaseShift"] = np.degrees(
            _column_to_array(csparc_data, "ctf/phase_shift_rad")
        )

    if "ctf/bfactor" in columns:
        relion_data["rlnCtfBfactor"] = _column_to_array(csparc_data, "ctf/bfactor")

    # Blobs / images
    if "blob/micrograph_blob/path" in columns:
        relion_data["rlnMicrographName"] = _column_to_array(
            csparc_data, "blob/micrograph_blob/path"
        )
    if "blob/path" in columns:
        image_name = _column_to_array(csparc_data, "blob/path")
        if "blob/idx" in columns:
            # RELION indexes into an image stack as 'index@path', starting at 1
            index = _column_to_array(csparc_data, "blob/idx") + 1
            image_name = np.asarray(
                [f"{i}@{p}" for i, p in zip(index, image_name)], dtype=object
            )
        relion_data["rlnImageName"] = image_name

    # Alignments
    if "alignments3D/pose" in columns:
        euler_angles = _convert_cs_pose_to_relion(
            _column_to_array(csparc_data, "alignments3D/pose")
        )
        relion_data["rlnAngleRot"] = np.asarray(euler_angles[:, 2])
        relion_data["rlnAngleTilt"] = np.asarray(euler_angles[:, 1])
        relion_data["rlnAnglePsi"] = np.asarray(euler_angles[:, 0])
    if "alignments3D/shift" in columns:
        # CryoSPARC shifts are in pixels, RELION origins are in angstroms
        shift = _column_to_array(csparc_data, "alignments3D/shift")
        shift_in_angstroms = shift if pixel_size is None else shift * pixel_size[:, None]
        relion_data["rlnOriginXAngst"] = shift_in_angstroms[:, 0]
        relion_data["rlnOriginYAngst"] = shift_in_angstroms[:, 1]

    return pd.DataFrame(relion_data)


def _split_optics_and_particle_data(
    relion_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split per-particle RELION columns into an optics block, with one row per
    optics group, and a particle block that points into it via 'rlnOpticsGroup'.
    """
    optics_columns = [
        column for column in RELION_OPTICS_COLUMNS if column in relion_data.columns
    ]
    particle_columns = [
        column for column in RELION_PARTICLE_COLUMNS if column in relion_data.columns
    ]
    if len(optics_columns) == 0:
        raise ValueError(
            "Could not build an optics group from the CryoSPARC file, because none "
            f"of the optics parameters {tuple(RELION_OPTICS_COLUMNS)} could be read "
            "from it."
        )
    # The optics block: one row per optics group, holding the parameters of the
    # group's first particle. Groups with equal parameters are kept separate, so
    # that the optics groups of the CryoSPARC file are preserved
    optics_data = (
        relion_data[["rlnOpticsGroup", *optics_columns]]
        .drop_duplicates(subset="rlnOpticsGroup")
        .sort_values("rlnOpticsGroup")
        .reset_index(drop=True)
    )
    # The particle block: one row per particle, pointing into the optics block
    particle_data = relion_data[[*particle_columns, "rlnOpticsGroup"]].copy()

    for data in (optics_data, particle_data):
        for column in ("rlnOpticsGroup", "rlnImageSize"):
            if column in data.columns:
                data[column] = data[column].astype("Int64")

    return optics_data, particle_data


def _column_to_array(dataframe: pd.DataFrame, column: str) -> np.ndarray:
    """Read a column as a `numpy` array, stacking columns that store
    multi-dimensional CryoSPARC fields as arrays.
    """
    values = dataframe[column].to_numpy()
    is_stacked_column = (
        values.dtype == object and values.size > 0 and isinstance(values[0], np.ndarray)
    )
    return np.stack(values) if is_stacked_column else values


def _convert_cs_pose_to_relion(pose_data):
    @eqx.filter_vmap
    def _quat_to_euler(quat):
        return convert_quaternion_to_euler_angles(quat, convention="zyz", extrinsic=True)

    quaternions = SO3.exp(pose_data).wxyz
    return _quat_to_euler(quaternions)
