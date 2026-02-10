"""
Check coverage with
pytest --cov-report term-missing:skip-covered --cov=src/cryojax/data/_relion tests/test_relion_dataset.py
"""  # noqa

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
from cryospax import RelionParticleDataset, RelionParticleParameterFile
from cryospax._dataset.relion import (
    _validate_starfile_data,
)
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


@pytest.fixture
def parameter_file(sample_starfile_path):
    return RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        options=dict(loads_envelope=True, loads_metadata=True),
    )


@pytest.fixture
def relion_parameters():
    image_config = cxs.BasicImageConfig(
        shape=(4, 4),
        pixel_size=1.5,
        voltage_in_kilovolts=300.0,
        padded_shape=(14, 14),
    )

    pose = cxs.EulerAnglePose()
    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.AstigmaticCTF(),
    )
    return dict(image_config=image_config, pose=pose, transfer_theory=transfer_theory)


#
# Tests for starfile loading


class TestErrorRaisingForLoading:
    def test_load_with_badparticle_name(self, parameter_file, sample_relion_project_path):
        parameter_file.particle_data.loc[0, "rlnImageName"] = "0.0"
        dataset = RelionParticleDataset(
            path_to_relion_project=sample_relion_project_path,
            parameter_file=parameter_file,
        )
        with pytest.raises(ValueError):
            dataset[0]

    def test_load_with_badparticle_name2(
        self, parameter_file, sample_relion_project_path
    ):
        parameter_file.particle_data.loc[0, "rlnImageName"] = "0000.mrcs"
        dataset = RelionParticleDataset(
            path_to_relion_project=sample_relion_project_path,
            parameter_file=parameter_file,
        )
        with pytest.raises(ValueError):
            dataset[0]

    def test_load_with_bad_shape(self, parameter_file, sample_relion_project_path):
        optics_data = parameter_file.optics_data
        optics_data.loc[0, "rlnImageSize"] = 1
        dataset = RelionParticleDataset(
            path_to_relion_project=sample_relion_project_path,
            parameter_file=parameter_file,
        )
        with pytest.raises(ValueError):
            dataset[0]

    def test_with_bad_indices(self, parameter_file, sample_relion_project_path):
        dataset = RelionParticleDataset(
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

    def test_validate_starfile_data(self):
        with pytest.raises(ValueError):
            _validate_starfile_data({"wrong": pd.DataFrame({})})

        with pytest.raises(ValueError):
            _validate_starfile_data({"particles": pd.DataFrame({})})

        mock_particles_df = pd.DataFrame(
            {
                "rlnDefocusU": 0.0,
                "rlnDefocusV": 0.0,
                "rlnDefocusAngle": 0.0,
                "rlnPhaseShift": 0.0,
                "rlnImageName": "mock.mrcs",
            },
            index=[0],
        )
        with pytest.raises(ValueError):
            _validate_starfile_data({"particles": mock_particles_df})

        with pytest.raises(ValueError):
            _validate_starfile_data(
                {"particles": mock_particles_df, "optics": pd.DataFrame({})}
            )


def test_make_image_config(sample_starfile_path):
    """Test the `make_image_config` argument to the parameter
    file."""
    # Test with a valid input

    make_fn = lambda s, ps, v: cxs.BasicImageConfig(s, ps, v, padded_shape=(22, 22))
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
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


def test_load_starfile_envelope_params(sample_starfile_path):
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        options=dict(loads_envelope=True, loads_metadata=False),
    )

    assert parameter_file.loads_envelope is True
    parameters = parameter_file[0]
    assert parameters["transfer_theory"].envelope is not None

    parameters = parameter_file[:]
    assert parameters["transfer_theory"].envelope is not None

    envelope = parameters["transfer_theory"].envelope

    particle_data = parameter_file.particle_data
    # check that envelope params match
    for i in range(len(parameter_file)):
        # check b-factors
        np.testing.assert_allclose(
            envelope.b_factor[i],  # type: ignore
            particle_data["rlnCtfBfactor"][i],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            envelope.amplitude[i],  # type: ignore
            particle_data["rlnCtfScalefactor"][i],
            rtol=1e-5,
        )
    return


def test_load_starfile_ctf_params(sample_starfile_path):
    is_shape = lambda shape, pytree: jax.tree.reduce(
        lambda x, y: x and y,
        jax.tree.map(lambda x: x.shape == shape, pytree),
    )

    def compute_defocus(defU, defV):
        return 0.5 * (defU + defV)

    def compute_astigmatism(defU, defV):
        return defU - defV

    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
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

    particle_data = parameter_file.particle_data
    # check CTF parameters
    for i in range(len(parameter_file)):
        # defocus
        np.testing.assert_allclose(
            ctf.defocus_in_angstroms[i],
            compute_defocus(
                particle_data["rlnDefocusU"][i],
                particle_data["rlnDefocusV"][i],
            ),
            rtol=1e-5,
        )

        # astigmatism
        np.testing.assert_allclose(
            ctf.astigmatism_in_angstroms[i],
            compute_astigmatism(
                particle_data["rlnDefocusU"][i],
                particle_data["rlnDefocusV"][i],
            ),
            rtol=1e-5,
        )

        # astigmatism_angle
        np.testing.assert_allclose(
            ctf.astigmatism_angle[i],
            particle_data["rlnDefocusAngle"][i],
            rtol=1e-5,
        )

        # phase shift
        np.testing.assert_allclose(
            transfer_theory.phase_shift[i],
            particle_data["rlnPhaseShift"][i],
            rtol=1e-5,
        )

    return


def test_load_starfile_pose_params(sample_starfile_path):
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        options=dict(loads_envelope=False, loads_metadata=True),
    )

    parameters = parameter_file[:]
    pose = parameters["pose"]

    particle_data = parameter_file.particle_data
    # Check pose parameters
    for i in range(len(parameter_file)):
        # offset x
        np.testing.assert_allclose(
            pose.offset_x_in_angstroms[i],
            -particle_data["rlnOriginXAngst"][i],  # conventions!
            rtol=1e-5,
        )

        # offset y
        np.testing.assert_allclose(
            pose.offset_y_in_angstroms[i],
            -particle_data["rlnOriginYAngst"][i],  # conventions!
            rtol=1e-5,
        )

        # phi angle - AngleRot
        np.testing.assert_allclose(
            pose.phi_angle[i],
            -particle_data["rlnAngleRot"][i],
            rtol=1e-5,
        )

        # theta angle - AngleTilt
        np.testing.assert_allclose(
            pose.theta_angle[i],
            -particle_data["rlnAngleTilt"][i],
            rtol=1e-5,
        )

        # psi angle - AnglePsi
        np.testing.assert_allclose(
            pose.psi_angle[i],
            -particle_data["rlnAnglePsi"][i],
            rtol=1e-5,
        )


def test_load_starfile_wo_metadata(sample_starfile_path):
    """Test loading a starfile without metadata."""
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        options=dict(loads_envelope=False, loads_metadata=False),
    )

    # check that metadata is empty dict
    assert "metadata" not in parameter_file[0]
    assert "metadata" not in parameter_file[:]
    assert not parameter_file.loads_metadata


def test_load_image_config_broadcasting(sample_starfile_path):
    """Test loading a starfile with optics group."""
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        options=dict(loads_envelope=False, loads_metadata=True),
    )
    parameters = parameter_file[:]
    image_config = parameters["image_config"]
    assert image_config.voltage_in_kilovolts.ndim == 0
    assert image_config.pixel_size.ndim == 0

    return


def test_parameter_file_setters(sample_starfile_path):
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        options=dict(
            loads_envelope=False,
            loads_metadata=False,
        ),
    )

    parameter_file.loads_metadata = True
    assert parameter_file.loads_metadata

    parameter_file.loads_envelope = True
    assert parameter_file.loads_envelope


def test_load_starfile_vs_mrcs_shape(sample_starfile_path, sample_relion_project_path):
    """Test loading a starfile with mrcs."""
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        options=dict(loads_envelope=False, loads_metadata=False),
    )
    dataset = RelionParticleDataset(parameter_file, sample_relion_project_path)

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


def test_no_load_parameters(sample_starfile_path, sample_relion_project_path):
    """Test loading a starfile with mrcs."""
    parameter_file = RelionParticleParameterFile(path_to_starfile=sample_starfile_path)
    dataset = RelionParticleDataset(parameter_file, sample_relion_project_path)

    # For particle stack with leading dim
    dataset.only_images = False
    particle_stack_params = dataset[:]
    dataset.only_images = True
    particle_stack_noparams = dataset[:]

    assert "parameters" not in particle_stack_noparams
    assert (
        particle_stack_params["images"].shape == particle_stack_noparams["images"].shape
    )
    assert isinstance(particle_stack_params["images"], np.ndarray)

    # For particle stack with no leading dim
    dataset.only_images = False
    particle_stack_params = dataset[0]
    dataset.only_images = True
    particle_stack_noparams = dataset[0]
    assert (
        particle_stack_params["images"].shape == particle_stack_noparams["images"].shape
    )
    assert isinstance(particle_stack_noparams["images"], np.ndarray)

    return


#
# Tests for starfile writing
#


@pytest.mark.parametrize(
    "index, loads_envelope",
    [
        (0, False),
        ([0, 1], False),
        (0, True),
    ],
)
def test_append_particle_parameters(index, loads_envelope):
    index = np.asarray(index)
    ndim = index.ndim

    @eqx.filter_vmap
    def make_particle_params(_):
        image_config = cxs.BasicImageConfig(
            shape=(4, 4),
            pixel_size=1.5,
            voltage_in_kilovolts=300.0,
        )

        pose = cxs.EulerAnglePose()
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.AstigmaticCTF(),
            envelope=im.FourierGaussian() if loads_envelope else None,
        )
        return dict(
            image_config=image_config,
            pose=pose,
            transfer_theory=transfer_theory,
        )

    # Make particle parameters, using custom metadata
    metadata = pd.DataFrame(
        data={
            "rlnMicrographName": list("dummy/micrograph.mrc" for _ in range(index.size)),
            "rlnCoordinateX": np.atleast_1d(np.full_like(index, 2, dtype=int)),
            "rlnCoordinateY": np.atleast_1d(np.full_like(index, 1, dtype=int)),
        },
    )
    particle_params = make_particle_params(jnp.atleast_1d(index))
    particle_params["metadata"] = metadata
    # ... custom metadata
    if ndim == 0:
        particle_params = jax.tree.map(
            lambda x: jnp.squeeze(x) if isinstance(x, jax.Array) else x, particle_params
        )
    # Add to dataset
    path_to_starfile = "tests/outputs/starfile_writing/test_particle_parameters.star"
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",
        exist_ok=True,
        options=dict(loads_envelope=loads_envelope, loads_metadata=False),
    )
    parameter_file.append(particle_params)
    # Make sure custom metadata was added
    particle_dataframe = parameter_file.particle_data
    assert set(metadata.columns).issubset(particle_dataframe.columns)
    # Make sure dataframes are the same
    metadata_extracted = particle_dataframe.loc[
        particle_dataframe.index[np.atleast_1d(index)],
        ["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"],
    ]
    np.testing.assert_equal(metadata.to_numpy(), metadata_extracted.to_numpy())
    # Make sure parameters read and the same as what was appended
    loaded_particle_params = parameter_file[index]
    del particle_params["metadata"]  # need to remove dataframe
    assert compare_dicts(loaded_particle_params, particle_params)


@pytest.mark.parametrize(
    "index, sets_envelope",
    [
        (0, False),
        ([0, 1], False),
        (0, False),
        (0, True),
    ],
)
def test_set_particle_parameters(
    sample_starfile_path,
    index,
    sets_envelope,
):
    index = np.asarray(index)
    n_particles, ndim = index.size, index.ndim

    def make_params(rng_key) -> dict:
        rng_keys = jr.split(rng_key, n_particles)
        make_pose = eqx.filter_vmap(
            lambda rng_key: cxs.EulerAnglePose.from_rotation(SO3.sample_uniform(rng_key))
        )
        pose = make_pose(rng_keys)
        return dict(
            image_config=cxs.BasicImageConfig(
                shape=(4, 4), pixel_size=3.324, voltage_in_kilovolts=121.3
            ),
            pose=pose,
            transfer_theory=cxs.ContrastTransferTheory(
                cxs.AstigmaticCTF(defocus_in_angstroms=1234.0),
                amplitude_contrast_ratio=0.1234,
                envelope=im.FourierGaussian(b_factor=12.34) if sets_envelope else None,
            ),
        )

    metadata = pd.DataFrame(
        data={
            "rlnMicrographName": list("dummy/micrograph.mrc" for _ in range(index.size)),
            "rlnCoordinateX": np.atleast_1d(np.full_like(index, 2, dtype=int)),
            "rlnCoordinateY": np.atleast_1d(np.full_like(index, 1, dtype=int)),
        },
    )
    rng_key = jr.key(0)
    new_parameters = make_params(rng_key)
    if ndim == 0:
        new_parameters = jax.tree.map(
            lambda x: jnp.squeeze(x) if isinstance(x, jax.Array) else x, new_parameters
        )
    new_parameters["metadata"] = metadata

    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        max_optics_groups=5,
        options=dict(loads_envelope=sets_envelope, loads_metadata=False),
    )
    # Set params
    with pytest.raises(ValueError):
        parameter_file[index] = new_parameters
    parameter_file.particle_data["rlnMicrographName"] = pd.Series(dtype=str)
    parameter_file.particle_data["rlnCoordinateX"] = pd.Series(dtype="Int64")
    parameter_file.particle_data["rlnCoordinateY"] = pd.Series(dtype="Int64")
    parameter_file[index] = new_parameters
    # Make sure custom metadata was added
    particle_dataframe = parameter_file.particle_data
    assert set(metadata.columns).issubset(particle_dataframe.columns)
    # Make sure dataframes are the same
    metadata_extracted = particle_dataframe.loc[
        particle_dataframe.index[np.atleast_1d(index)],
        ["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"],
    ]
    np.testing.assert_equal(metadata.to_numpy(), metadata_extracted.to_numpy())
    # Load params that were just set
    loaded_parameters = parameter_file[index]
    del new_parameters["metadata"]
    assert compare_dicts(new_parameters, loaded_parameters)


def test_file_exists_error():
    # Create pytrees
    parameters = dict(
        image_config=cxs.BasicImageConfig(
            shape=(4, 4), pixel_size=1.1, voltage_in_kilovolts=300.0
        ),
        pose=cxs.EulerAnglePose(),
        transfer_theory=cxs.ContrastTransferTheory(ctf=cxs.AstigmaticCTF()),
    )
    # Add to dataset
    path_to_starfile = "tests/outputs/starfile_writing/test_particle_parameters.star"
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",
        exist_ok=True,
    )
    parameter_file.append(parameters)
    parameter_file.save(overwrite=True)

    # Test no exist_ok
    with pytest.raises(FileExistsError):
        _ = RelionParticleParameterFile(
            path_to_starfile=path_to_starfile,
            mode="w",
            exist_ok=False,
        )
    # Clean up
    shutil.rmtree(parameter_file.path_to_output.parent)


def test_file_not_found_error():
    dummy_path_to_starfile = "path/to/nonexistant/dir/nonexistant_file.star"

    # Test no exist_ok
    with pytest.raises(FileNotFoundError):
        _ = RelionParticleParameterFile(
            path_to_starfile=dummy_path_to_starfile,
            mode="r",
        )

    return


def test_set_wrong_parameters_error():
    # Wrong parameters
    wrong_pose = cxs.QuaternionPose()
    wrong_transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.AstigmaticCTF(), envelope=im.FourierDC()
    )
    # Right parameters
    right_pose = cxs.EulerAnglePose()
    right_transfer_theory = cxs.ContrastTransferTheory(ctf=cxs.AstigmaticCTF())
    image_config = cxs.BasicImageConfig(
        shape=(4, 4), pixel_size=1.1, voltage_in_kilovolts=300.0
    )
    # Create pytrees
    wrong_parameters_1 = dict(
        image_config=image_config,
        pose=right_pose,
        transfer_theory=wrong_transfer_theory,
    )
    temp = dict(
        image_config=image_config,
        pose=right_pose,
        transfer_theory=right_transfer_theory,
    )
    wrong_parameters_2 = eqx.tree_at(lambda x: x["pose"], temp, wrong_pose)
    # Now the parameter dataset
    # Add to dataset
    path_to_starfile = "path/to/dummy/project/and/starfile.star"
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",
        exist_ok=True,
    )

    with pytest.raises(ValueError):
        parameter_file.append(wrong_parameters_1)

    with pytest.raises(TypeError):
        parameter_file.append(wrong_parameters_2)


def test_bad_pytree_error():
    # Right parameters
    make_pose = eqx.filter_vmap(
        lambda x, y, phi, theta, psi: cxs.EulerAnglePose(x, y, phi, theta, psi)
    )
    pose = make_pose(
        jnp.atleast_1d(1.0),
        jnp.atleast_1d(-1.0),
        jnp.atleast_1d(1.0),
        jnp.atleast_1d(2.0),
        jnp.atleast_1d(3.0),
    )
    pose = eqx.tree_at(
        lambda x: x.offset_in_angstroms,
        pose,
        jnp.stack(2 * [np.asarray((1.0, 2.0))], axis=-1),
    )
    transfer_theory = cxs.ContrastTransferTheory(ctf=cxs.AstigmaticCTF())
    image_config = cxs.BasicImageConfig(
        shape=(4, 4), pixel_size=1.1, voltage_in_kilovolts=300.0
    )
    # Create pytrees
    parameters = dict(
        image_config=image_config,
        pose=pose,
        transfer_theory=transfer_theory,
    )
    # Now the parameter dataset
    # Add to dataset
    path_to_starfile = "path/to/dummy/project/and/starfile.star"
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",
        exist_ok=True,
    )

    with pytest.raises(ValueError):
        parameter_file.append(parameters)


def test_write_image(
    sample_relion_project_path,
    sample_starfile_path,
    relion_parameters,
):
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        exist_ok=True,
        max_optics_groups=10,
    )
    parameter_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )

    dataset = RelionParticleDataset(
        parameter_file,
        path_to_relion_project=sample_relion_project_path,
        mode="w",
    )
    particle_data = dataset.parameter_file.particle_data
    assert particle_data["rlnImageName"].isna().all()

    shape = relion_parameters["image_config"].shape
    particle = dict(
        parameters=relion_parameters,
        images=jnp.zeros(shape, dtype=np.float32),
    )
    bad_shape_particle = dict(
        parameters=relion_parameters,
        images=jnp.zeros((shape[0], shape[1] + 1), dtype=np.float32),
    )
    bad_dim_particle = eqx.tree_at(
        lambda x: x["images"], bad_shape_particle, jnp.zeros(shape[0], dtype=np.float32)
    )

    with pytest.raises(ValueError):
        dataset[0] = bad_shape_particle

    with pytest.raises((ValueError, TypeCheckError)):
        dataset[0] = bad_dim_particle

    with pytest.raises(IOError):
        dataset[0] = particle

    dataset.mrcfile_options = dict(prefix="f", overwrite=True)
    dataset[0] = particle

    particle_data = dataset.parameter_file.particle_data
    rln_image_name = particle_data["rlnImageName"][0]
    # Assert entry was written
    assert not pd.isna(rln_image_name)
    assert particle_data["rlnImageName"][1:].isna().all()
    # Assert file was written and delete it
    filename = str(rln_image_name).split("@")[1]
    path_to_filename = os.path.join(sample_relion_project_path, filename)
    assert os.path.exists(path_to_filename)
    os.remove(path_to_filename)
    assert not os.path.exists(path_to_filename)


def test_write_particle_batched_particle_parameters():
    @partial(eqx.filter_vmap, in_axes=(0), out_axes=eqx.if_array(0))
    def _make_particle_params(dummy_idx):
        image_config = cxs.BasicImageConfig(
            shape=(4, 4),
            pixel_size=1.5,
            voltage_in_kilovolts=300.0,
        )

        pose = cxs.EulerAnglePose()
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.AstigmaticCTF(), envelope=im.FourierGaussian()
        )
        return {
            "image_config": image_config,
            "pose": pose,
            "transfer_theory": transfer_theory,
        }

    particle_params = _make_particle_params(jnp.array([0, 0, 0, 0, 0]))
    new_parameters_file = RelionParticleParameterFile.empty(
        path_to_starfile="dummy.star", exist_ok=True, num_particles=0, max_optics_groups=1
    )

    new_parameters_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )

    new_parameters_file.append(particle_params)
    new_parameters_file.save(overwrite=True)
    # and try to save again
    with pytest.raises(FileExistsError):
        new_parameters_file.save(overwrite=False)

    parameter_file = RelionParticleParameterFile.load(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        loads_envelope=True,
        loads_metadata=False,
    )

    loaded_params = parameter_file[:]
    compare_dicts(loaded_params, particle_params)
    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


def test_write_starfile_different_envs():
    def _make_particle_params(envelope):
        image_config = cxs.BasicImageConfig(
            shape=(4, 4),
            pixel_size=1.5,
            voltage_in_kilovolts=300.0,
        )

        pose = cxs.EulerAnglePose()
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.AstigmaticCTF(),
            envelope=envelope,
        )
        return {
            "image_config": image_config,
            "pose": pose,
            "transfer_theory": transfer_theory,
        }

    particle_params = _make_particle_params(im.FourierGaussian())
    new_parameters_file = RelionParticleParameterFile.empty(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        exist_ok=True,
        num_particles=0,
        max_optics_groups=1,
    )
    new_parameters_file.append(particle_params)
    new_parameters_file.save(overwrite=True)

    particle_params = _make_particle_params(im.FourierConstant(1.0))
    new_parameters_file = RelionParticleParameterFile.empty(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        exist_ok=True,
        num_particles=0,
        max_optics_groups=1,
    )
    new_parameters_file.append(particle_params)
    new_parameters_file.save(overwrite=True)

    particle_params = _make_particle_params(None)
    new_parameters_file = RelionParticleParameterFile.empty(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        exist_ok=True,
        num_particles=0,
        max_optics_groups=1,
    )
    new_parameters_file.append(particle_params)
    new_parameters_file.save(overwrite=True)

    with pytest.raises(ValueError):
        particle_params = _make_particle_params(im.FourierDC(1.0))
        new_parameters_file = RelionParticleParameterFile.empty(
            path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
            exist_ok=True,
            num_particles=0,
            max_optics_groups=1,
        )
        new_parameters_file.append(particle_params)
        new_parameters_file.save(overwrite=True)

    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


def test_raise_errors_parameter_file(sample_starfile_path):
    from jaxtyping import TypeCheckError

    with pytest.raises((ValueError, TypeCheckError)):
        parameter_file = RelionParticleParameterFile(
            path_to_starfile=sample_starfile_path,
            mode="CRYOEM",  # type: ignore
            options=dict(loads_envelope=False, loads_metadata=False),
        )

    parameter_file = RelionParticleParameterFile.load(
        path_to_starfile=sample_starfile_path,
        loads_envelope=False,
        loads_metadata=False,
    )

    assert parameter_file.mode == "r"

    # now set to write mode and try to filter
    with pytest.raises(ValueError):
        parameter_file = RelionParticleParameterFile(
            path_to_starfile=sample_starfile_path,
            mode="w",
            exist_ok=True,
            options=dict(loads_envelope=False, loads_metadata=False),
            selection_filter={"rlnAngleRot": lambda x: x < 1000.0},
        )


def test_raise_errors_stack_dataset(sample_starfile_path, sample_relion_project_path):
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        options=dict(loads_envelope=False, loads_metadata=False),
    )
    parameter_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )

    # remove "rlnImageName" column
    particle_data = parameter_file.particle_data
    parameter_file._starfile_data["particles"] = particle_data.drop(
        columns=["rlnImageName"]
    )

    with pytest.raises(IOError):
        particle_dataset = RelionParticleDataset(
            parameter_file,
            path_to_relion_project=sample_relion_project_path,
            mode="r",
        )

    # Now set to write mode

    parameter_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )
    particle_dataset = RelionParticleDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="w",
        mrcfile_options={"overwrite": False},
    )

    parameters = parameter_file[0]
    image_shape = parameters["image_config"].shape

    particle_stack = {
        "parameters": parameters,
        "images": jnp.zeros(image_shape, dtype=np.float32),
    }

    with pytest.raises(ValueError):
        particle_dataset[np.array([0])] = particle_stack

    with pytest.raises(TypeError):
        particle_dataset[0] = "dummy"  # type: ignore

    with pytest.raises(TypeError):
        particle_dataset.append("dummy")  # type: ignore

    with pytest.raises(ValueError):
        particle_dataset.append({"parameters": None, "images": particle_stack["images"]})

    with pytest.raises(ValueError):
        particle_dataset.write_images(
            index_array=np.array([0, 1], dtype=int), images=np.zeros((100, 10, 10))
        )

    with pytest.raises(ValueError):
        particle_dataset.write_images(
            index_array=np.array([0, 1], dtype=int), images=np.zeros((10, *image_shape))
        )

    # and clean
    shutil.rmtree("tests/outputs/starfile_writing/")


def test_append_relion_stack_dataset():
    @partial(eqx.filter_vmap, in_axes=(0), out_axes=eqx.if_array(0))
    def _make_particle_params(dummy_idx):
        image_config = cxs.BasicImageConfig(
            shape=(4, 4),
            pixel_size=1.5,
            voltage_in_kilovolts=300.0,
        )

        pose = cxs.EulerAnglePose()
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.AstigmaticCTF(), envelope=im.FourierGaussian()
        )
        return {
            "image_config": image_config,
            "pose": pose,
            "transfer_theory": transfer_theory,
        }

    new_stack = RelionParticleDataset.empty(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        path_to_relion_project="tests/outputs/starfile_writing/",
        max_optics_groups=1,
        exist_ok=True,
        num_particles=0,
        mrcfile_options={"overwrite": False},
    )

    n_images = 10
    particle_params = _make_particle_params(jnp.ones(n_images))
    shape = particle_params["image_config"].shape
    images = jax.random.normal(
        jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
    )

    new_stack.append(
        {
            "parameters": particle_params,
            "images": images,
        }
    )

    # clean up
    shutil.rmtree("tests/outputs/starfile_writing/")
    return
