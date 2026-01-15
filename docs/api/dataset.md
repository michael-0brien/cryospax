# Cryo-EM dataset manipulation in RELION and cryoSPARC

CryoSPAX implements interfaces for reading/writing to common cryo-EM software frameworks, such as RELION and cryoSPARC.

??? abstract "`cryospax.AbstractDataset`"
    ::: cryospax.AbstractDataset
        options:
            members:
                - __init__
                - __getitem__

### Reading/writing parameter files (e.g. STAR files)

??? abstract "`cryospax.AbstractParticleParameterFile`"
    ::: cryospax.AbstractParticleParameterFile
        options:
            members:
                - __init__

??? abstract "`cryospax.AbstractRelionParticleParameterFile`"
    ::: cryospax.AbstractRelionParticleParameterFile
        options:
            members:
                - __init__

::: cryospax.RelionParticleParameterFile
    options:
        members:
            - __init__
            - __getitem__
            - __setitem__
            - append
            - starfile_data

### Datasets: parameter and image manipulation

??? abstract "`cryospax.AbstractParticleDataset`"
    ::: cryospax.AbstractParticleDataset
        options:
            members:
                - __init__
                - __getitem__
                - __setitem__
                - append

::: cryospax.RelionParticleDataset
    options:
        members:
            - __init__
            - __getitem__
            - __setitem__
            - append
            - write_images
