# Overview

## What does cryoSPAX implement?

### Cryo-EM dataset manipulation

CryoSPAX includes tools for manipulating datasets in existing single particle analysis frameworks, such as RELION.

```python
import cryospax as spx

# Instantiate RELION dataset
dataset = spx.RelionParticleDataset.load(
    path_to_starfile="./path/to/particles.star", path_to_relion_project="./path/to/project/"
)
# Load first 10 images and particle parameters into `cryojax` classes
particle_info = dataset[0:10]
images, parameters = particle_info["images"], particle_info["parameters"]
```
