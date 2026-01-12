# Cryo-EM dataset manipulation in RELION and cryoSPARC

CryoSPAX implements interfaces for reading/writing to common cryo-EM software frameworks, such as RELION and cryoSPARC.

## Reading/writing parameter files (e.g. STAR files)

??? abstract "`cryospax.AbstractParticleStarfile`"
    ::: cryospax.AbstractParticleStarfile

::: cryospax.RelionParticleParameterFile
    options:
        members:
            - __init__
            - starfile_data
            - __getitem__
            - __setitem__
            - append

## Datasets: parameter and image manipulation

??? abstract "`cryospax.AbstractParticleDataset`"
    ::: cryospax.AbstractParticleDataset

::: cryospax.RelionParticleDataset
    options:
        members:
            - __init__
            - __getitem__
            - __setitem__
            - append
            - write_images
