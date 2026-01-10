# Cryo-EM dataset manipulation in RELION and cryoSPARC

CryoJAX SPA implements interfaces for reading/writing to common cryo-EM software frameworks, such as RELION and cryoSPARC.

## Reading/writing parameter files (e.g. STAR files)

??? abstract "`cryojax_spa.AbstractParticleStarfile`"
    ::: cryojax_spa.AbstractParticleStarfile

::: cryojax_spa.RelionParticleParameterFile
    options:
        members:
            - __init__
            - starfile_data
            - __getitem__
            - __setitem__
            - append

## Datasets: parameter and image manipulation

??? abstract "`cryojax_spa.AbstractParticleDataset`"
    ::: cryojax_spa.AbstractParticleDataset

::: cryojax_spa.RelionParticleDataset
    options:
        members:
            - __init__
            - __getitem__
            - __setitem__
            - append
            - write_images
