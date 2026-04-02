import cryospax as spx
import equinox as eqx
import pytest


@pytest.fixture
def registered_datasets(
    sample_starfile_path, sample_csfile_path, sample_relion_project_path
):
    rl_parameter_file = spx.RelionParticleParameterFile(sample_starfile_path)
    cs_parameter_file = spx.CryoSparcParticleParameterFile(sample_csfile_path)

    return [
        rl_parameter_file,
        spx.RelionParticleDataset(
            rl_parameter_file.copy(),
            sample_relion_project_path,
            only_images=False,
        ),
        spx.RelionParticleDataset(
            rl_parameter_file.copy(),
            sample_relion_project_path,
            only_images=True,
        ),
        cs_parameter_file,
        spx.CryoSparcParticleDataset(
            cs_parameter_file.copy(),
            sample_relion_project_path,
            only_images=False,
        ),
        spx.CryoSparcParticleDataset(
            cs_parameter_file.copy(),
            sample_relion_project_path,
            only_images=True,
        ),
    ]


@pytest.fixture
def erroneous_datasets(
    sample_starfile_path, sample_csfile_path, sample_relion_project_path
):
    rl_parameter_file = spx.RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path, options=dict(loads_metadata=True)
    )
    cs_parameter_file = spx.CryoSparcParticleParameterFile(
        path_to_csfile=sample_csfile_path, options=dict(loads_metadata=True)
    )
    return [
        rl_parameter_file,
        spx.RelionParticleDataset(
            rl_parameter_file.copy(),
            sample_relion_project_path,
        ),
        cs_parameter_file,
        spx.CryoSparcParticleDataset(
            cs_parameter_file.copy(),
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
