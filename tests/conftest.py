import os

import jax
import pytest
from cryojax.io import read_array_from_mrc


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def sample_starfile_path():
    return os.path.join(os.path.dirname(__file__), "data", "test_starfile.star")


@pytest.fixture
def sample_relion_project_path():
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def sample_image_stack_path(sample_relion_project_path):
    return os.path.join(sample_relion_project_path, "img_00000.mrcs")


@pytest.fixture
def sample_image_stack(sample_image_stack_path):
    return read_array_from_mrc(sample_image_stack_path)
