import pathlib
from typing import cast

import equinox as eqx
from cryojax.simulator import AxisAnglePose, EulerAnglePose

from .csparc import CryoSparcParticleParameterFile
from .relion import (
    RelionParticleParameterFile,
    _format_number_for_filename,
)


def convert_csparc_to_relion(
    cs_parameter_file: CryoSparcParticleParameterFile,
    path_to_starfile: str | pathlib.Path,
    exists_ok: bool = False,
) -> RelionParticleParameterFile:
    """Convert a `CryoSparcParticleParameterFile` to a
    `RelionParticleParameterFile`.

    **Arguments:**
    - `cs_parameter_file`:
        The `CryoSparcParticleParameterFile` to convert.
    - `path_to_starfile`:
        The path to the new starfile.
    - `exists_ok`:
        Parameter to instantiate `RelionParticleParameterFile`.

    **Returns:**
    A `RelionParticleParameterFile` containing the converted particle
    parameters from the input `CryoSparcParticleParameterFile`.

    **Notes:**
    This does not create a starfile at `path_to_starfile`. To create the starfile, call
    `relion_particle_parameter_file.save(overwrite=...)`.

    """

    relion_particle_parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",
        exist_ok=exists_ok,
        options={
            "loads_envelope": cs_parameter_file.loads_envelope,
        },
    )

    # set particle parameters
    parameters = cs_parameter_file[:]
    parameters["pose"] = _convert_axisangle_to_euler(parameters["pose"])
    parameters = cast(dict[str, object], parameters)
    relion_particle_parameter_file.append(parameters)

    particle_filenames = cs_parameter_file.csfile_data["blob/path"].astype(str)
    particle_indices = cs_parameter_file.csfile_data["blob/idx"]

    rln_image_names = [
        _format_number_for_filename(int(i + 1), n_characters=6)
        + "@"
        + particle_filenames[i]
        for i in particle_indices
    ]

    # set image names
    relion_particle_parameter_file.starfile_data["particles"]["rlnImageName"] = (
        rln_image_names
    )

    return relion_particle_parameter_file


@eqx.filter_vmap
def _convert_axisangle_to_euler(pose: AxisAnglePose) -> EulerAnglePose:
    return EulerAnglePose.from_rotation_and_translation(
        rotation=pose.rotation, offset_in_angstroms=pose.offset_in_angstroms
    )
