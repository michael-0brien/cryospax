import os
import shutil
from functools import partial
from typing import cast

import cryojax.ndimage as im
import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import pytest
from cryojax.rotations import SO3
from cryospax import (
    CryoSparcParticleDataset,
    CryoSparcParticleParameterFile,
    RelionParticleDataset,
    RelionParticleParameterFile,
    convert_csparc_to_relion,
)
from cryospax._dataset.csparc import _validate_csfile_data
from jaxtyping import TypeCheckError


def compare_dicts(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    bool_arrays, bool_others = [], []
    for k in dict1.keys():
        arrays1, others1 = eqx.partition(dict1[k], eqx.is_array)
        arrays2, others2 = eqx.partition(dict2[k], eqx.is_array)
        bool_arrays.append(
            jnp.allclose(arr1, arr2)  # type: ignore
            for arr1, arr2 in zip(jax.tree.flatten(arrays1), jax.tree.flatten(arrays2))
        )
        bool_others.append(
            other1 == other2  # type: ignore
            for other1, other2 in zip(
                jax.tree.flatten(others1), jax.tree.flatten(others2)
            )
        )
    return all(bool_arrays) and all(bool_others)


@eqx.filter_vmap(in_axes=(0, 0))
def compare_pose_rotations(pose1, pose2):
    return eqx.tree_equal(pose1.rotation.wxyz, pose2.rotation.wxyz, rtol=1e-6)


@pytest.fixture
def parameter_file(sample_csfile_path):
    return CryoSparcParticleParameterFile(
        path_to_csfile=sample_csfile_path,
        options=dict(loads_envelope=True, loads_metadata=True),
    )


@pytest.fixture
def cs_parameters():
    image_config = cxs.BasicImageConfig(
        shape=(4, 4),
        pixel_size=1.5,
        voltage_in_kilovolts=300.0,
        padded_shape=(14, 14),
    )

    pose = cxs.AxisAnglePose()
    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.AstigmaticCTF(),
    )
    return dict(image_config=image_config, pose=pose, transfer_theory=transfer_theory)


#
# Tests for csfile loading


class TestErrorRaisingForLoading:
    def test_load_with_badparticle_name(self, parameter_file, sample_relion_project_path):
        parameter_file.csfile_data.loc[0, "blob/path"] = "0.0"
        dataset = CryoSparcParticleDataset(
            path_to_relion_project=sample_relion_project_path,
            parameter_file=parameter_file,
        )
        with pytest.raises(FileNotFoundError):
            dataset[0]

        def test_load_with_badparticle_name2(
            self, parameter_file, sample_relion_project_path
        ):
            parameter_file.csfile_data.loc[0, "blob/path"] = "0000.mrcs"
            dataset = CryoSparcParticleDataset(
                path_to_relion_project=sample_relion_project_path,
                parameter_file=parameter_file,
            )
            with pytest.raises(TypeError):
                dataset[0]

    def test_load_with_bad_shape(self, parameter_file, sample_relion_project_path):
        parameter_file.csfile_data.at[0, "blob/shape"] = [1, 1]
        dataset = CryoSparcParticleDataset(
            path_to_relion_project=sample_relion_project_path,
            parameter_file=parameter_file,
        )
        with pytest.raises(ValueError):
            dataset[0]

    def test_with_bad_indices(self, parameter_file, sample_relion_project_path):
        dataset = CryoSparcParticleDataset(
            path_to_relion_project=sample_relion_project_path,
            parameter_file=parameter_file,
        )

        # overflow index
        with pytest.raises(IndexError):
            parameter_file[len(parameter_file)]

        with pytest.raises(IndexError):
            dataset[len(dataset)]

        # overflow slice
        with pytest.raises(IndexError):
            parameter_file[len(parameter_file) :]

        with pytest.raises(IndexError):
            dataset[len(dataset) :]

        # wrong index type
        with pytest.raises(IndexError):
            parameter_file["wrong_index"]

        with pytest.raises(IndexError):
            dataset["wrong_index"]  # type: ignore


def test_make_image_config(sample_csfile_path):
    """Test the `make_image_config` argument to the parameter
    file."""
    # Test with a valid input

    make_fn = lambda s, ps, v: cxs.BasicImageConfig(s, ps, v, padded_shape=(22, 22))
    parameter_file = CryoSparcParticleParameterFile(
        path_to_csfile=sample_csfile_path,
        options=dict(
            loads_envelope=True,
            loads_metadata=True,
            make_image_config=make_fn,
        ),
    )
    image_config = parameter_file[0]["image_config"]

    ref_config = make_fn((16, 16), 12.0, 300.0)
    assert image_config.shape == ref_config.shape
    assert image_config.pixel_size == np.asarray(ref_config.pixel_size)
    assert image_config.voltage_in_kilovolts == np.asarray(
        ref_config.voltage_in_kilovolts
    )

    assert image_config.padded_shape == ref_config.padded_shape


def test_load_csfile_envelope_params(sample_csfile_path):
    parameter_file = CryoSparcParticleParameterFile(
        path_to_csfile=sample_csfile_path,
        options=dict(loads_envelope=True, loads_metadata=False),
    )

    assert parameter_file.loads_envelope is True
    parameters = parameter_file[0]
    assert parameters["transfer_theory"].envelope is not None

    parameters = parameter_file[:]
    assert parameters["transfer_theory"].envelope is not None

    envelope = parameters["transfer_theory"].envelope

    particle_data = parameter_file.csfile_data
    # check that envelope params match
    for i in range(len(parameter_file)):
        # check b-factors
        np.testing.assert_allclose(
            envelope.b_factor[i],  # type: ignore
            particle_data["ctf/bfactor"][i],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            envelope.amplitude[i],  # type: ignore
            particle_data["ctf/scale"][i],
            rtol=1e-5,
        )
    return


def test_load_csfile_ctf_params(sample_csfile_path):
    is_shape = lambda shape, pytree: jax.tree.reduce(
        lambda x, y: x and y,
        jax.tree.map(lambda x: x.shape == shape, pytree),
    )

    def compute_defocus(defU, defV):
        return 0.5 * (defU + defV)

    def compute_astigmatism(defU, defV):
        return defU - defV

    parameter_file = CryoSparcParticleParameterFile(
        path_to_csfile=sample_csfile_path,
        options=dict(loads_envelope=False, loads_metadata=True),
    )

    assert parameter_file.loads_envelope is False

    # Check loading 1 particle
    parameters = parameter_file[0]
    assert parameters["transfer_theory"].envelope is None
    assert is_shape((), eqx.filter(parameters["transfer_theory"].ctf, eqx.is_array))

    # Check loading >1 particles
    parameters = parameter_file[:]
    assert parameters["transfer_theory"].envelope is None
    assert is_shape(
        (len(parameter_file),),
        eqx.filter(parameters["transfer_theory"].ctf, eqx.is_array),
    )

    transfer_theory = parameters["transfer_theory"]
    ctf = cast(cxs.AstigmaticCTF, transfer_theory.ctf)

    particle_data = parameter_file.csfile_data
    # check CTF parameters
    for i in range(len(parameter_file)):
        # defocus
        np.testing.assert_allclose(
            ctf.defocus_in_angstroms[i],
            compute_defocus(
                particle_data["ctf/df1_A"][i],
                particle_data["ctf/df2_A"][i],
            ),
            rtol=1e-5,
        )

        # astigmatism
        np.testing.assert_allclose(
            ctf.astigmatism_in_angstroms[i],
            compute_astigmatism(
                particle_data["ctf/df1_A"][i],
                particle_data["ctf/df2_A"][i],
            ),
            rtol=1e-5,
        )

        # astigmatism_angle
        np.testing.assert_allclose(
            np.deg2rad(ctf.astigmatism_angle[i]),
            particle_data["ctf/df_angle_rad"][i],
            rtol=1e-5,
        )

        # phase shift
        np.testing.assert_allclose(
            np.deg2rad(transfer_theory.phase_shift[i]),
            particle_data["ctf/phase_shift_rad"][i],
            rtol=1e-5,
        )

    return


def test_load_csfile_pose_params(sample_csfile_path):
    parameter_file = CryoSparcParticleParameterFile(
        path_to_csfile=sample_csfile_path,
        options=dict(loads_envelope=False, loads_metadata=True),
    )

    particle_data = parameter_file.csfile_data
    pixel_size = particle_data["blob/psize_A"][0]
    # Check pose parameters
    for i in range(len(parameter_file)):
        pose = parameter_file[i]["pose"]
        # offset
        np.testing.assert_allclose(
            pose.offset_in_angstroms,
            -particle_data["alignments3D/shift"][i] * pixel_size,  # conventions!
            rtol=1e-5,
        )

        # rotation
        np.testing.assert_allclose(
            np.deg2rad(pose.to_inverse_rotation().euler_vector),
            particle_data["alignments3D/pose"][i],
            rtol=1e-5,
        )


def test_load_csfile_wo_metadata(sample_csfile_path):
    """Test loading a csfile without metadata."""
    parameter_file = CryoSparcParticleParameterFile(
        path_to_csfile=sample_csfile_path,
        options=dict(loads_envelope=False, loads_metadata=False),
    )

    # check that metadata is empty dict
    assert "metadata" not in parameter_file[0]
    assert "metadata" not in parameter_file[:]
    assert not parameter_file.loads_metadata


def test_load_image_config_broadcasting(sample_csfile_path):
    """Test loading a csfile with optics group."""
    parameter_file = CryoSparcParticleParameterFile(
        path_to_csfile=sample_csfile_path,
        options=dict(loads_envelope=False, loads_metadata=True),
    )
    parameters = parameter_file[:]
    image_config = parameters["image_config"]
    assert image_config.voltage_in_kilovolts.ndim == 0
    assert image_config.pixel_size.ndim == 0

    return


def test_parameter_file_setters(sample_csfile_path):
    parameter_file = CryoSparcParticleParameterFile(
        path_to_csfile=sample_csfile_path,
        options=dict(
            loads_envelope=False,
            loads_metadata=False,
        ),
    )

    parameter_file.loads_metadata = True
    assert parameter_file.loads_metadata

    parameter_file.loads_envelope = True
    assert parameter_file.loads_envelope


def test_load_csfile_vs_mrcs_shape(sample_csfile_path, sample_relion_project_path):
    """Test loading a csfile with mrcs."""
    parameter_file = CryoSparcParticleParameterFile(
        path_to_csfile=sample_csfile_path,
        options=dict(loads_envelope=False, loads_metadata=False),
    )
    dataset = CryoSparcParticleDataset(parameter_file, sample_relion_project_path)

    particle_info = dataset[:]
    parameter_info = particle_info["parameters"]
    image_config = parameter_info["image_config"]
    assert particle_info["images"].shape == (
        len(parameter_file),
        *image_config.shape,
    )

    particle_stack = dataset[0]
    image_config = parameter_info["image_config"]
    assert particle_stack["images"].shape == image_config.shape

    particle_stack = dataset[0:2]
    image_config = parameter_info["image_config"]
    assert particle_stack["images"].shape == (2, *image_config.shape)

    assert len(dataset) == len(parameter_file)

    return


def test_no_load_parameters(sample_csfile_path, sample_relion_project_path):
    """Test loading a csfile with mrcs."""
    parameter_file = CryoSparcParticleParameterFile(path_to_csfile=sample_csfile_path)
    dataset = CryoSparcParticleDataset(parameter_file, sample_relion_project_path)

    # For particle stack with leading dim
    dataset.just_images = False
    particle_stack_params = dataset[:]
    dataset.just_images = True
    particle_stack_noparams = dataset[:]

    assert "parameters" not in particle_stack_noparams
    assert (
        particle_stack_params["images"].shape == particle_stack_noparams["images"].shape
    )
    assert isinstance(particle_stack_params["images"], np.ndarray)

    # For particle stack with no leading dim
    dataset.just_images = False
    particle_stack_params = dataset[0]
    dataset.just_images = True
    particle_stack_noparams = dataset[0]
    assert (
        particle_stack_params["images"].shape == particle_stack_noparams["images"].shape
    )
    assert isinstance(particle_stack_noparams["images"], np.ndarray)

    return


def test_matches_with_relion(
    sample_csfile_path, sample_starfile_path, sample_relion_project_path
):
    """Test that loading a csfile and starfile with the same data gives the same
    parameters."""

    parameter_file_cs = CryoSparcParticleParameterFile(sample_csfile_path)
    parameter_file_rl = RelionParticleParameterFile(sample_starfile_path)

    dataset_cs = CryoSparcParticleDataset(
        parameter_file=parameter_file_cs,
        path_to_relion_project=sample_relion_project_path,
    )

    dataset_rl = RelionParticleDataset(
        parameter_file=parameter_file_rl,
        path_to_relion_project=sample_relion_project_path,
    )

    params_cs = parameter_file_cs[:]
    params_rl = parameter_file_rl[:]

    # Check parameters
    assert eqx.tree_equal(
        params_cs["transfer_theory"], params_rl["transfer_theory"], rtol=1e-6
    ), "Transfer theory parameters do not match"

    assert eqx.tree_equal(
        params_cs["image_config"], params_rl["image_config"], rtol=1e-6
    ), "Image config parameters do not match"

    assert eqx.tree_equal(
        params_cs["pose"].offset_in_angstroms,
        params_rl["pose"].offset_in_angstroms,
        rtol=1e-6,
    ), "Pose offset parameters do not match"

    assert compare_pose_rotations(params_cs["pose"], params_rl["pose"]).all(), (
        "Pose rotation parameters do not match"
    )

    # check images
    images_cs = dataset_cs[:]["images"]
    images_rl = dataset_rl[:]["images"]

    assert jnp.allclose(images_cs, images_rl), "Particle images do not match"

@pytest.mark.parametrize("loads_envelope", [True, False])
def test_conversion(sample_csfile_path, loads_envelope):

    parameter_file_cs = CryoSparcParticleParameterFile(
        sample_csfile_path, options=dict(loads_envelope=loads_envelope)
    )
    parameter_file_rl = convert_csparc_to_relion(
        parameter_file_cs, "tests/outputs/test.star", exists_ok=True
    )

    params_cs = parameter_file_cs[:]
    params_rl = parameter_file_rl[:]

    # Check parameters
    assert eqx.tree_equal(
        params_cs["transfer_theory"], params_rl["transfer_theory"], rtol=1e-6
    ), "Transfer theory parameters do not match"

    assert eqx.tree_equal(
        params_cs["image_config"], params_rl["image_config"], rtol=1e-6
    ), "Image config parameters do not match"

    assert eqx.tree_equal(
        params_cs["pose"].offset_in_angstroms,
        params_rl["pose"].offset_in_angstroms,
        rtol=1e-6,
    ), "Pose offset parameters do not match"

    assert compare_pose_rotations(params_cs["pose"], params_rl["pose"]).all(), (
        "Pose rotation parameters do not match"
    )
