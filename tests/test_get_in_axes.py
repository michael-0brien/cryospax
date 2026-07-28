import cryospax as spx
import equinox as eqx
import pytest


@pytest.fixture
def registered_datasets(sample_starfile_path, sample_relion_project_path):
    parameter_file = spx.RelionParticleParameterFile(sample_starfile_path)
    return [
        parameter_file,
        spx.RelionParticleDataset(
            parameter_file.copy(),
            sample_relion_project_path,
            only_images=False,
        ),
        spx.RelionParticleDataset(
            parameter_file.copy(),
            sample_relion_project_path,
            only_images=True,
        ),
    ]


@pytest.fixture
def erroneous_datasets(sample_starfile_path, sample_relion_project_path):
    parameter_file = spx.RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path, options=dict(loads_metadata=True)
    )
    return [
        parameter_file,
        spx.RelionParticleDataset(
            parameter_file.copy(),
            sample_relion_project_path,
        ),
    ]


def test_vmap(registered_datasets):
    fn = lambda _x: _x

    for dataset in registered_datasets:
        fn_vmap = eqx.filter_vmap(fn, in_axes=(spx.get_in_axes(dataset),))
        element = dataset[0:1]
        _ = fn_vmap(element)


def test_vmap_error(erroneous_datasets):
    for dataset in erroneous_datasets:
        with pytest.raises(AttributeError):
            _ = spx.get_in_axes(dataset)
