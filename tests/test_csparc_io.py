"""Tests for CryoSPARC '.cs' file serialization, i.e. `cryospax.read_csparc_file`
and `cryospax.read_csparc_file_as_starfile`.
"""

import pathlib
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest
from cryojax.rotations import SO3
from cryospax import (
    RelionParticleParameterFile,
    read_csparc_file,
    read_csparc_file_as_starfile,
    write_starfile,
)
from cryospax._dataset.relion import _validate_starfile_data
from cryospax._io._csparc import (
    RELION_OPTICS_COLUMNS,
    RELION_PARTICLE_COLUMNS,
)


def write_csfile(path: str | pathlib.Path, **fields) -> pathlib.Path:
    """Write a CryoSPARC-style '.cs' file, i.e. a `numpy` structured array of
    named fields, from `fields` given as `{'blob/psize_A': array, ...}`.
    """
    arrays = {key: np.asarray(value) for key, value in fields.items()}
    num_particles = len(next(iter(arrays.values())))
    dtype = [
        ((key, value.dtype, value.shape[1:]) if value.ndim > 1 else (key, value.dtype))
        for key, value in arrays.items()
    ]
    csparc_array = np.zeros(num_particles, dtype=dtype)
    for key, value in arrays.items():
        csparc_array[key] = value
    # `np.save` appends a '.npy' suffix unless it is passed a file object
    path = pathlib.Path(path)
    with open(path, "wb") as file:
        np.save(file, csparc_array)
    return path


def make_csparc_fields(num_particles: int = 3, **overrides) -> dict:
    """A minimal set of CryoSPARC fields that `read_csparc_file_as_starfile`
    can fully convert, i.e. one that covers every optics and particle column.
    """
    index = np.arange(num_particles)
    fields = {
        "uid": np.arange(1, num_particles + 1, dtype="<u8"),
        "blob/path": np.array(
            [f"J1/imgs/{i:04d}.mrcs".encode() for i in index], dtype="S32"
        ),
        "blob/idx": index.astype("<u4"),
        "blob/shape": np.full((num_particles, 2), 8, dtype="<u4"),
        "blob/psize_A": np.full(num_particles, 1.5, dtype="<f4"),
        "blob/micrograph_blob/path": np.array(
            [f"J1/mics/{i:04d}.mrc".encode() for i in index], dtype="S32"
        ),
        "ctf/accel_kv": np.full(num_particles, 300.0, dtype="<f4"),
        "ctf/cs_mm": np.full(num_particles, 2.7, dtype="<f4"),
        "ctf/amp_contrast": np.full(num_particles, 0.1, dtype="<f4"),
        "ctf/df1_A": (10000.0 + 100.0 * index).astype("<f4"),
        "ctf/df2_A": (10050.0 + 100.0 * index).astype("<f4"),
        "ctf/df_angle_rad": (0.1 * index).astype("<f4"),
        "ctf/phase_shift_rad": (0.01 * index).astype("<f4"),
        "ctf/bfactor": index.astype("<f4"),
        "alignments3D/shift": np.stack(
            [index.astype("<f4"), -index.astype("<f4")], axis=-1
        ),
        "alignments3D/pose": np.stack([0.1 + 0.1 * index] * 3, axis=-1).astype("<f4"),
    }
    fields.update(overrides)
    return fields


@pytest.fixture
def csfile_path(tmp_path):
    return write_csfile(tmp_path / "particles.cs", **make_csparc_fields())


@pytest.fixture
def passthrough_path(tmp_path):
    """A passthrough file for `csfile_path`, which repeats one of its fields
    and adds one of its own.
    """
    fields = make_csparc_fields()
    return write_csfile(
        tmp_path / "passthrough_particles.cs",
        uid=fields["uid"],
        # Repeated from the main file, with a different value so that the
        # merge can be checked
        **{"blob/psize_A": np.full(3, 99.0, dtype="<f4")},
        **{"location/micrograph_uid": np.arange(10, 13, dtype="<u8")},
    )


@pytest.fixture
def unique_uid_csfile_path(tmp_path, sample_csfile_path):
    """The sample '.cs' file, whose 'uid' entries are all zero, with unique
    'uid' entries so that it can be converted to STAR file data.
    """
    csparc_array = np.load(sample_csfile_path)
    csparc_array = csparc_array.copy()
    csparc_array["uid"] = np.arange(1, len(csparc_array) + 1)
    path = tmp_path / "sample_with_unique_uid.cs"
    with open(path, "wb") as file:
        np.save(file, csparc_array)
    return path


#
# Tests for reading '.cs' files
#


class TestReadCsparcFile:
    def test_reads_all_csparc_fields(self, sample_csfile_path):
        csparc_array = np.load(sample_csfile_path)
        csparc_data = read_csparc_file(sample_csfile_path)

        assert list(csparc_data.columns) == list(csparc_array.dtype.names)  # type: ignore
        assert len(csparc_data) == len(csparc_array)

    def test_decodes_byte_strings(self, csfile_path):
        csparc_data = read_csparc_file(csfile_path)

        assert csparc_data["blob/path"].iloc[0] == "J1/imgs/0000.mrcs"
        assert csparc_data["blob/micrograph_blob/path"].iloc[0] == "J1/mics/0000.mrc"

    def test_stores_multidimensional_fields_as_arrays(self, csfile_path):
        csparc_data = read_csparc_file(csfile_path)

        shape = csparc_data["blob/shape"].iloc[0]
        assert isinstance(shape, np.ndarray)
        np.testing.assert_array_equal(shape, [8, 8])
        assert np.stack(csparc_data["alignments3D/pose"].to_numpy()).shape == (3, 3)

    def test_reads_scalar_fields_without_copying_values(self, csfile_path):
        fields = make_csparc_fields()
        csparc_data = read_csparc_file(csfile_path)

        np.testing.assert_allclose(csparc_data["ctf/df1_A"], fields["ctf/df1_A"])
        np.testing.assert_allclose(csparc_data["blob/psize_A"], fields["blob/psize_A"])


class TestReadCsparcFileWithPassthrough:
    def test_merges_passthrough_fields(self, csfile_path, passthrough_path):
        csparc_data = read_csparc_file(csfile_path, passthrough_path)

        assert len(csparc_data) == 3
        assert "location/micrograph_uid" in csparc_data.columns
        np.testing.assert_array_equal(
            csparc_data["location/micrograph_uid"], [10, 11, 12]
        )

    def test_keeps_fields_of_main_file_when_repeated(self, csfile_path, passthrough_path):
        csparc_data = read_csparc_file(csfile_path, passthrough_path)

        # The passthrough file stores 99.0, the main file stores 1.5
        np.testing.assert_allclose(csparc_data["blob/psize_A"], 1.5)
        # ... and no '_x'/'_y' columns are generated by the merge
        assert not any("_x" in column or "_y" in column for column in csparc_data.columns)

    def test_merges_on_uid_not_on_row_order(self, tmp_path, csfile_path):
        """The passthrough file is merged on the particle 'uid', so its rows may
        be in a different order than those of the main file.
        """
        shuffled_uid = np.array([3, 1, 2], dtype="<u8")
        passthrough_path = write_csfile(
            tmp_path / "shuffled_passthrough.cs",
            uid=shuffled_uid,
            **{"location/micrograph_uid": shuffled_uid * 10},
        )
        csparc_data = read_csparc_file(csfile_path, passthrough_path)

        np.testing.assert_array_equal(
            csparc_data["location/micrograph_uid"], csparc_data["uid"] * 10
        )

    def test_keeps_only_particles_in_both_files(self, tmp_path, csfile_path):
        passthrough_path = write_csfile(
            tmp_path / "subset_passthrough.cs",
            uid=np.array([2, 3, 4], dtype="<u8"),
            **{"location/micrograph_uid": np.arange(3, dtype="<u8")},
        )
        csparc_data = read_csparc_file(csfile_path, passthrough_path)

        np.testing.assert_array_equal(csparc_data["uid"], [2, 3])


class TestReadCsparcFileErrors:
    def test_error_with_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_csparc_file(tmp_path / "does_not_exist.cs")

    def test_error_with_wrong_suffix(self, tmp_path, csfile_path):
        wrong_suffix_path = tmp_path / "particles.star"
        wrong_suffix_path.write_bytes(csfile_path.read_bytes())
        with pytest.raises(OSError, match="'.cs'"):
            read_csparc_file(wrong_suffix_path)

    def test_error_with_wrong_passthrough_suffix(self, tmp_path, csfile_path):
        wrong_suffix_path = tmp_path / "passthrough_particles.star"
        wrong_suffix_path.write_bytes(csfile_path.read_bytes())
        with pytest.raises(OSError, match="'.cs'"):
            read_csparc_file(csfile_path, wrong_suffix_path)

    def test_error_with_unstructured_array(self, tmp_path):
        path = tmp_path / "not_a_csfile.cs"
        with open(path, "wb") as file:
            np.save(file, np.zeros((3, 3)))
        with pytest.raises(OSError, match="structured array"):
            read_csparc_file(path)


#
# Tests for converting '.cs' files to STAR file data
#


class TestConvertToStarfileData:
    def test_returns_optics_and_particle_blocks(self, csfile_path):
        starfile_data = read_csparc_file_as_starfile(csfile_path)

        assert set(starfile_data.keys()) == {"optics", "particles"}
        assert len(starfile_data["particles"]) == 3
        # This is the point of the conversion: the result must be loadable as
        # RELION STAR file data
        _validate_starfile_data(starfile_data)

    def test_converts_only_supported_columns(self, csfile_path):
        starfile_data = read_csparc_file_as_starfile(csfile_path)

        assert list(starfile_data["optics"].columns) == [
            "rlnOpticsGroup",
            *RELION_OPTICS_COLUMNS,
        ]
        assert list(starfile_data["particles"].columns) == [
            *RELION_PARTICLE_COLUMNS,
            "rlnOpticsGroup",
        ]

    def test_drops_uid(self, csfile_path):
        starfile_data = read_csparc_file_as_starfile(csfile_path)

        assert "uid" not in starfile_data["particles"].columns
        assert "uid" not in starfile_data["optics"].columns

    def test_integer_columns_have_integer_dtypes(self, csfile_path):
        starfile_data = read_csparc_file_as_starfile(csfile_path)

        assert starfile_data["optics"]["rlnOpticsGroup"].dtype == "Int64"
        assert starfile_data["optics"]["rlnImageSize"].dtype == "Int64"
        assert starfile_data["particles"]["rlnOpticsGroup"].dtype == "Int64"

    def test_converts_optics_parameters(self, csfile_path):
        optics_data = read_csparc_file_as_starfile(csfile_path)["optics"]

        assert len(optics_data) == 1
        optics_group = optics_data.iloc[0]
        assert optics_group["rlnOpticsGroup"] == 1
        np.testing.assert_allclose(optics_group["rlnImagePixelSize"], 1.5)
        np.testing.assert_allclose(optics_group["rlnVoltage"], 300.0)
        np.testing.assert_allclose(optics_group["rlnSphericalAberration"], 2.7)
        np.testing.assert_allclose(optics_group["rlnAmplitudeContrast"], 0.1)
        assert optics_group["rlnImageSize"] == 8

    def test_converts_ctf_parameters(self, csfile_path):
        fields = make_csparc_fields()
        particle_data = read_csparc_file_as_starfile(csfile_path)["particles"]

        np.testing.assert_allclose(particle_data["rlnDefocusU"], fields["ctf/df1_A"])
        np.testing.assert_allclose(particle_data["rlnDefocusV"], fields["ctf/df2_A"])
        np.testing.assert_allclose(particle_data["rlnCtfBfactor"], fields["ctf/bfactor"])
        # CryoSPARC stores angles in radians, RELION in degrees
        np.testing.assert_allclose(
            particle_data["rlnDefocusAngle"],
            np.degrees(fields["ctf/df_angle_rad"]),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            particle_data["rlnPhaseShift"],
            np.degrees(fields["ctf/phase_shift_rad"]),
            rtol=1e-6,
        )

    def test_converts_defocus_from_alternative_field_names(self, tmp_path):
        """CryoSPARC writes defocus as either 'ctf/df1_A' or 'ctf/defocus_u',
        depending on the job that produced the file.
        """
        fields = make_csparc_fields()
        fields["ctf/defocus_u"] = fields.pop("ctf/df1_A")
        fields["ctf/defocus_v"] = fields.pop("ctf/df2_A")
        fields["ctf/defocus_angle"] = fields.pop("ctf/df_angle_rad")
        path = write_csfile(tmp_path / "alternative_names.cs", **fields)
        particle_data = read_csparc_file_as_starfile(path)["particles"]

        np.testing.assert_allclose(particle_data["rlnDefocusU"], fields["ctf/defocus_u"])
        np.testing.assert_allclose(particle_data["rlnDefocusV"], fields["ctf/defocus_v"])
        np.testing.assert_allclose(
            particle_data["rlnDefocusAngle"],
            np.degrees(fields["ctf/defocus_angle"]),
            rtol=1e-6,
        )

    def test_converts_image_names_to_relion_stack_index(self, csfile_path):
        particle_data = read_csparc_file_as_starfile(csfile_path)["particles"]

        # RELION indexes into an image stack as 'index@path', starting at 1,
        # while CryoSPARC's 'blob/idx' starts at 0
        assert list(particle_data["rlnImageName"]) == [
            "1@J1/imgs/0000.mrcs",
            "2@J1/imgs/0001.mrcs",
            "3@J1/imgs/0002.mrcs",
        ]
        assert list(particle_data["rlnMicrographName"]) == [
            "J1/mics/0000.mrc",
            "J1/mics/0001.mrc",
            "J1/mics/0002.mrc",
        ]

    def test_converts_image_names_without_stack_index(self, tmp_path):
        fields = make_csparc_fields()
        del fields["blob/idx"]
        path = write_csfile(tmp_path / "no_index.cs", **fields)
        particle_data = read_csparc_file_as_starfile(path)["particles"]

        assert list(particle_data["rlnImageName"]) == [
            "J1/imgs/0000.mrcs",
            "J1/imgs/0001.mrcs",
            "J1/imgs/0002.mrcs",
        ]

    def test_converts_shifts_to_angstroms(self, csfile_path):
        fields = make_csparc_fields()
        particle_data = read_csparc_file_as_starfile(csfile_path)["particles"]

        # CryoSPARC shifts are in pixels, RELION origins are in angstroms
        shift_in_angstroms = (
            fields["alignments3D/shift"] * fields["blob/psize_A"][:, None]
        )
        np.testing.assert_allclose(
            particle_data["rlnOriginXAngst"], shift_in_angstroms[:, 0], rtol=1e-6
        )
        np.testing.assert_allclose(
            particle_data["rlnOriginYAngst"], shift_in_angstroms[:, 1], rtol=1e-6
        )

    def test_leaves_shifts_in_pixels_without_a_pixel_size(self, tmp_path):
        fields = make_csparc_fields()
        del fields["blob/psize_A"]
        path = write_csfile(tmp_path / "no_pixel_size.cs", **fields)
        particle_data = read_csparc_file_as_starfile(path)["particles"]

        np.testing.assert_allclose(
            particle_data["rlnOriginXAngst"], fields["alignments3D/shift"][:, 0]
        )

    def test_writes_angles_in_degrees(self, csfile_path):
        particle_data = read_csparc_file_as_starfile(csfile_path)["particles"]

        for column in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"):
            assert (particle_data[column].abs() <= 360.0).all()
        # A pose of zero is the identity rotation, i.e. zero Euler angles
        fields = make_csparc_fields()
        fields["alignments3D/pose"] = np.zeros((3, 3), dtype="<f4")
        path = write_csfile(csfile_path.parent / "identity_pose.cs", **fields)
        particle_data = read_csparc_file_as_starfile(path)["particles"]
        for column in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"):
            np.testing.assert_allclose(particle_data[column], 0.0)

    def test_converts_poses_to_relion_convention(self, tmp_path, csfile_path):
        """A pose read back from the converted STAR file must be the rotation
        stored in the '.cs' file as an axis-angle vector.

        CryoSPARC stores the rotation of the *frame*, so the pose loaded by
        cryospax is its inverse. This is the convention of the sample STAR file,
        and the one that `CryoSparcParticleParameterFile` used to load.
        """
        fields = make_csparc_fields()
        starfile_data = read_csparc_file_as_starfile(csfile_path)
        path_to_starfile = tmp_path / "converted_poses.star"
        write_starfile(starfile_data, path_to_starfile)
        pose = RelionParticleParameterFile(path_to_starfile)[:]["pose"]

        expected_rotation = SO3.exp(jnp.asarray(fields["alignments3D/pose"])).inverse()
        np.testing.assert_allclose(
            np.asarray(pose.rotation.as_matrix()),
            np.asarray(expected_rotation.as_matrix()),
            rtol=1e-5,
            atol=1e-6,
        )


class TestOpticsGroups:
    def test_groups_identical_optics_parameters(self, csfile_path):
        starfile_data = read_csparc_file_as_starfile(csfile_path)

        assert len(starfile_data["optics"]) == 1
        np.testing.assert_array_equal(starfile_data["particles"]["rlnOpticsGroup"], 1)

    def test_splits_distinct_optics_parameters(self, tmp_path):
        fields = make_csparc_fields(num_particles=4)
        # Two pixel sizes and two voltages, in three unique combinations
        fields["blob/psize_A"] = np.array([1.5, 3.0, 1.5, 1.5], dtype="<f4")
        fields["ctf/accel_kv"] = np.array([300.0, 300.0, 200.0, 300.0], dtype="<f4")
        path = write_csfile(tmp_path / "many_optics_groups.cs", **fields)
        starfile_data = read_csparc_file_as_starfile(path)

        optics_data, particle_data = starfile_data["optics"], starfile_data["particles"]
        assert len(optics_data) == 3
        # Groups are numbered from one, in order of first appearance
        np.testing.assert_array_equal(optics_data["rlnOpticsGroup"], [1, 2, 3])
        np.testing.assert_array_equal(particle_data["rlnOpticsGroup"], [1, 2, 3, 1])
        # ... and every particle's optics parameters are those of its group
        optics_of_particle = particle_data[["rlnOpticsGroup"]].merge(
            optics_data, on="rlnOpticsGroup", how="left"
        )
        np.testing.assert_allclose(
            optics_of_particle["rlnImagePixelSize"], fields["blob/psize_A"]
        )
        np.testing.assert_allclose(
            optics_of_particle["rlnVoltage"], fields["ctf/accel_kv"]
        )


class TestConvertToStarfileDataErrors:
    def test_error_with_duplicate_uid(self, sample_csfile_path):
        # Every 'uid' of the sample file is zero, so particles cannot be
        # assigned to an optics group
        with pytest.raises(ValueError, match="duplicate 'uid'"):
            read_csparc_file_as_starfile(sample_csfile_path)

    def test_error_with_non_square_images(self, tmp_path):
        fields = make_csparc_fields()
        fields["blob/shape"] = np.array([[8, 4], [8, 4], [8, 4]], dtype="<u4")
        path = write_csfile(tmp_path / "non_square.cs", **fields)
        with pytest.raises(ValueError, match="Non-square"):
            read_csparc_file_as_starfile(path)

    def test_error_without_optics_parameters(self, tmp_path):
        fields = make_csparc_fields()
        for key in (
            "blob/psize_A",
            "blob/shape",
            "ctf/accel_kv",
            "ctf/cs_mm",
            "ctf/amp_contrast",
        ):
            del fields[key]
        path = write_csfile(tmp_path / "no_optics.cs", **fields)
        with pytest.raises(ValueError, match="optics group"):
            read_csparc_file_as_starfile(path)


#
# Tests against a reference STAR file holding the same particles as the
# sample '.cs' file
#


class TestMatchesReferenceStarfile:
    @pytest.fixture
    def converted_parameter_file(self, tmp_path, unique_uid_csfile_path):
        starfile_data = read_csparc_file_as_starfile(unique_uid_csfile_path)
        path_to_starfile = tmp_path / "converted.star"
        write_starfile(starfile_data, path_to_starfile)
        return RelionParticleParameterFile(
            path_to_starfile, options=dict(loads_envelope=True)
        )

    @pytest.fixture
    def reference_parameter_file(self, sample_starfile_path):
        return RelionParticleParameterFile(
            sample_starfile_path, options=dict(loads_envelope=True)
        )

    def test_num_particles(self, converted_parameter_file, reference_parameter_file):
        assert len(converted_parameter_file) == len(reference_parameter_file)

    def test_optics_parameters(self, converted_parameter_file, reference_parameter_file):
        converted, reference = (
            converted_parameter_file.optics_data,
            reference_parameter_file.optics_data,
        )
        for column in RELION_OPTICS_COLUMNS:
            np.testing.assert_allclose(
                converted[column].astype(float),
                reference[column].astype(float),
                rtol=1e-6,
                err_msg=f"Optics column '{column}' does not match",
            )

    def test_ctf_parameters(self, converted_parameter_file, reference_parameter_file):
        converted = converted_parameter_file[:]["transfer_theory"]
        reference = reference_parameter_file[:]["transfer_theory"]

        np.testing.assert_allclose(
            np.asarray(converted.ctf.defocus_in_angstroms),
            np.asarray(reference.ctf.defocus_in_angstroms),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(converted.ctf.astigmatism_in_angstroms),
            np.asarray(reference.ctf.astigmatism_in_angstroms),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(converted.ctf.astigmatism_angle),
            np.asarray(reference.ctf.astigmatism_angle),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(converted.phase_shift), np.asarray(reference.phase_shift)
        )

    def test_pose_offsets(self, converted_parameter_file, reference_parameter_file):
        np.testing.assert_allclose(
            np.asarray(converted_parameter_file[:]["pose"].offset_in_angstroms),
            np.asarray(reference_parameter_file[:]["pose"].offset_in_angstroms),
            rtol=1e-6,
        )

    def test_image_names(self, converted_parameter_file, reference_parameter_file):
        def split_image_name(image_name):
            index, _, path = image_name.partition("@")
            return int(index), path

        converted = converted_parameter_file.particle_data["rlnImageName"]
        reference = reference_parameter_file.particle_data["rlnImageName"]
        # RELION zero pads the stack index, the conversion does not
        assert [split_image_name(name) for name in converted] == [
            split_image_name(name) for name in reference
        ]

    def test_pose_rotations(self, converted_parameter_file, reference_parameter_file):
        np.testing.assert_allclose(
            np.asarray(converted_parameter_file[:]["pose"].rotation.as_matrix()),
            np.asarray(reference_parameter_file[:]["pose"].rotation.as_matrix()),
            rtol=1e-5,
            atol=1e-6,
        )


#
# Tests for the 'cryospax csparc2spax' program
#


def run_csparc2spax(*arguments):
    path_to_program = (
        pathlib.Path(__file__).parents[1] / "cryospax" / "programs" / "csparc2spax.py"
    )
    return subprocess.run(
        [sys.executable, str(path_to_program), *map(str, arguments)],
        capture_output=True,
        text=True,
    )


class TestCsparc2Spax:
    def test_writes_starfile(self, tmp_path, csfile_path):
        path_to_starfile = tmp_path / "particles.star"
        result = run_csparc2spax(csfile_path, path_to_starfile)

        assert result.returncode == 0, result.stderr
        assert path_to_starfile.exists()
        parameter_file = RelionParticleParameterFile(path_to_starfile)
        assert len(parameter_file) == 3

    def test_writes_starfile_with_passthrough(
        self, tmp_path, csfile_path, passthrough_path
    ):
        path_to_starfile = tmp_path / "particles.star"
        result = run_csparc2spax(
            csfile_path, path_to_starfile, "--passthrough", passthrough_path
        )

        assert result.returncode == 0, result.stderr
        parameter_file = RelionParticleParameterFile(path_to_starfile)
        assert len(parameter_file) == 3

    def test_does_not_overwrite_without_flag(self, tmp_path, csfile_path):
        path_to_starfile = tmp_path / "particles.star"
        assert run_csparc2spax(csfile_path, path_to_starfile).returncode == 0

        result = run_csparc2spax(csfile_path, path_to_starfile)
        assert result.returncode != 0
        assert "already exists" in result.stderr

        result = run_csparc2spax(csfile_path, path_to_starfile, "--overwrite")
        assert result.returncode == 0, result.stderr

    def test_reports_conversion_errors_without_a_traceback(
        self, tmp_path, sample_csfile_path
    ):
        # The sample file has duplicate 'uid' entries
        result = run_csparc2spax(sample_csfile_path, tmp_path / "particles.star")

        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "duplicate 'uid'" in result.stderr

    def test_help_documents_that_this_is_not_a_general_converter(self):
        result = run_csparc2spax("--help")

        assert result.returncode == 0
        assert "general purpose" in result.stdout
        assert "RelionParticleParameterFile" in result.stdout
