import cryospax as spx
import equinox as eqx
import pytest


@pytest.fixture
def registered_datasets(sample_starfile_path, sample_relion_project_path):
    return [
        spx.RelionParticleParameterFile(sample_starfile_path),
        spx.RelionParticleDataset(
            spx.RelionParticleParameterFile(sample_starfile_path),
            sample_relion_project_path,
        ),
    ]


@pytest.fixture
def erroneous_datasets(sample_starfile_path):
    return [
        spx.RelionParticleParameterFile(
            path_to_starfile=sample_starfile_path, options=dict(loads_metadata=True)
        )
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
