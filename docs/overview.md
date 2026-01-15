# Overview

CryoSPAX is built to support scientists looking to create of custom single particle analysis workflows that leverage [cryoJAX](https://github.com/michael-0brien/cryojax) at scale. It is a *small package* that seeks to simplify cryo-EM research; it does not provide end-to-end single particle analysis workflows.

## What does cryoSPAX implement?

### Cryo-EM dataset manipulation

CryoSPAX includes tools for manipulating datasets in existing single particle analysis frameworks, such as RELION.

```python
import cryospax as spa

# Instantiate RELION dataset
parameter_file = spa.RelionParticleParameterFile(path_to_starfile="./path/to/particles.star")
dataset = spa.RelionParticleDataset(parameter_file, path_to_relion_project="./path/to/project/")
# Load first 10 images and particle parameters into `cryojax` classes
particle_info = dataset[0:10]
images, parameters = particle_info["images"], particle_info["parameters"]
```
