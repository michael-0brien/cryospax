import shutil
from functools import partial

import cryojax.ndimage as im
import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cryospax import (
    RelionParticleDataset,
    RelionParticleParameterFile,
    simulate_particle_stack,
)


def test_write_simulated_image_stack_from_starfile_jit(sample_starfile_path):
    def _mock_compute_image(a, b, per_particle_args):
        # Mock the image computation
        return per_particle_args

    """Test writing a simulated image stack from a starfile."""
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        loads_envelope=False,
        loads_metadata=False,
    )
    parameter_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )

    n_images = len(parameter_file)
    shape = parameter_file[0]["image_config"].shape
    true_images = jax.random.normal(
        jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
    )

    # Create a simulated image stack
    new_stack = RelionParticleDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="w",
        mrcfile_settings={"overwrite": True},
    )

    simulate_particle_stack(
        new_stack,
        simulate_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=true_images,
        is_jittable=True,
        overwrite=True,
    )

    # try to overwrite
    simulate_particle_stack(
        new_stack,
        simulate_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=true_images,
        is_jittable=True,
        overwrite=True,
    )

    # Now trigger overwrite error
    with pytest.raises(FileExistsError):
        simulate_particle_stack(
            new_stack,
            simulate_fn=_mock_compute_image,
            constant_args=(1.0, 2.0),
            per_particle_args=true_images,
            is_jittable=True,
            overwrite=False,
        )

    # load the simulated image stack
    particle_dataset = RelionParticleDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="r",
    )

    images = particle_dataset[:]["images"]
    np.testing.assert_allclose(
        images,
        true_images.astype(jnp.float32),
    )

    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


def test_write_simulated_image_stack_from_starfile_nojit(sample_starfile_path):
    def _mock_compute_image(particle_parameters, constant_args, per_particle_args):
        # Mock the image computation
        c1, c2 = constant_args
        image = per_particle_args
        return image / jnp.linalg.norm(image)

    """Test writing a simulated image stack from a starfile."""
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        loads_envelope=False,
        loads_metadata=False,
    )
    parameter_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )

    n_images = len(parameter_file)
    shape = parameter_file[0]["image_config"].shape
    true_images = jax.random.normal(
        jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
    )

    # Create a simulated image stack
    new_stack = RelionParticleDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="w",
        mrcfile_settings={"overwrite": True},
    )

    simulate_particle_stack(
        new_stack,
        simulate_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=true_images,
        overwrite=True,
    )

    particle_dataset = RelionParticleDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="r",
    )
    images = particle_dataset[:]["images"]
    assert jnp.allclose(
        images,
        true_images / np.linalg.norm(true_images, axis=(1, 2), keepdims=True),
    )

    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


def test_write_single_image(sample_starfile_path):
    def _mock_compute_image(particle_parameters, constant_args, per_particle_args):
        # Mock the image computation
        c1, c2 = constant_args
        p1, p2 = per_particle_args
        image = jnp.ones(particle_parameters["image_config"].shape, dtype=jnp.float32)
        return image / jnp.linalg.norm(image)

    selection_filter = {
        "rlnImageName": lambda x: np.where(x == "0000001@000000.mrcs", True, False)
    }
    """Test writing a simulated image stack from a starfile."""
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        loads_envelope=False,
        loads_metadata=False,
        selection_filter=selection_filter,
    )
    parameter_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )

    # Create a simulated image stack
    new_stack = RelionParticleDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="w",
        mrcfile_settings={"overwrite": True},
    )

    n_images = 1
    simulate_particle_stack(
        new_stack,
        simulate_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=(3.0 * jnp.ones(n_images), 4.0 * jnp.ones(n_images)),
        overwrite=True,
        images_per_file=1,
    )

    particle_dataset = RelionParticleDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="r",
    )
    images = particle_dataset[:]["images"]
    np.testing.assert_allclose(
        images,
        np.ones_like(images) / np.linalg.norm(np.ones_like(images)),
    )

    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


def test_load_multiple_mrcs():
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

    def _mock_compute_image(a, b, per_particle_args):
        # Mock the image computation
        return per_particle_args

    particle_params = _make_particle_params(jnp.ones(10))
    parameters_file = RelionParticleParameterFile(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        mode="w",
        exists_ok=True,
        updates_optics_group=True,
        loads_envelope=True,
    )
    parameters_file.append(particle_params)

    n_images = len(parameters_file)
    # print(f"Number of images: {n_images}")
    shape = parameters_file[0]["image_config"].shape
    true_images = jax.random.normal(
        jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
    )

    new_dataset = RelionParticleDataset(
        parameters_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="w",
        mrcfile_settings={"overwrite": True},
    )

    # Create a simulated image stack
    simulate_particle_stack(
        new_dataset,
        simulate_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=true_images,
        overwrite=True,
        images_per_file=3,
        batch_size=3,
    )

    n_tests = 10
    for _ in range(n_tests):
        indices = np.random.choice(len(parameters_file), size=3, replace=False)

        images = new_dataset[indices]["images"]
        np.testing.assert_allclose(
            images,
            true_images[indices],
        )

    shutil.rmtree("tests/outputs/starfile_writing/")
    return
