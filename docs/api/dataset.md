# Cryo-EM dataset manipulation

CryoSPAX implements interfaces for reading/writing to common cryo-EM software frameworks, such as RELION and cryoSPARC.

??? abstract "`cryospax.AbstractDataset`"
    ::: cryospax.AbstractDataset
        options:
            members:
                - __init__
                - __len__
                - __getitem__

---

Datasets output a pytree that can be passed the JAX transformations. A particular challenge is passing these pytrees to `jax.vmap`; different cryoSPAX datasets may load different pytrees, and each pytree may have different arrays which are broadcasted.

::: cryospax.get_in_axes

## Reading/writing parameter files (e.g. STAR files)

??? abstract "`cryospax.AbstractParticleParameterFile`"
    ::: cryospax.AbstractParticleParameterFile
        options:
            members:
                - __init__
                - __getitem__
                - __setitem__
                - __len__
                - append
                - save
                - path_to_output
                - mode

??? abstract "`cryospax.AbstractRelionParticleParameterFile`"
    ::: cryospax.AbstractRelionParticleParameterFile
        options:
            group_by_category: false
            members:
                - __init__
                - __getitem__
                - __setitem__
                - __len__
                - append
                - save
                - path_to_starfile
                - starfile_data
                - loads_metadata
                - loads_envelope
                - updates_optics_group
                - make_image_config

::: cryospax.RelionParticleParameterFile
    options:
        group_by_category: false
        members:
            - __init__
            - __getitem__
            - __setitem__
            - __len__
            - append
            - copy
            - save
            - starfile_data
            - path_to_starfile
            - path_to_output
            - mode
            - loads_metadata
            - loads_envelope
            - updates_optics_group
            - make_image_config

## Datasets: parameter and image manipulation

??? abstract "`cryospax.AbstractParticleDataset`"
    ::: cryospax.AbstractParticleDataset
        options:
            members:
                - __init__
                - __getitem__
                - __setitem__
                - __len__
                - append
                - parameter_file
                - write_images
                - mode
                - only_images

::: cryospax.RelionParticleDataset
    options:
        group_by_category: false
        members:
            - __init__
            - __getitem__
            - __setitem__
            - __len__
            - append
            - write_images
            - parameter_file
            - path_to_relion_project
            - mrcfile_settings
            - only_images

## Basic I/O

::: cryospax.read_starfile

::: cryospax.write_starfile
