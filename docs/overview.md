# Overview

CryoSPAX is built to support applications of [cryoJAX](https://github.com/michael-0brien/cryojax) for building cryo-EM single particle analysis workflows and algorithms---and deploying them at scale. It is a *small package* that seeks to simplify new cryo-EM research; it does not provide end-to-end single particle analysis workflows.

## What does cryoSPAX implement?

### Cryo-EM dataset manipulation

CryoSPAX includes tools for manipulating datasets in existing single particle analysis frameworks, such as RELION.

```python
import cryospax as spx

# Instantiate RELION dataset
parameter_file = spx.RelionParticleParameterFile(path_to_starfile="./path/to/particles.star")
dataset = spx.RelionParticleDataset(parameter_file, path_to_relion_project="./path/to/project/")
# Load first 10 images and particle parameters into `cryojax` classes
particle_info = dataset[0:10]
images, parameters = particle_info["images"], particle_info["parameters"]
```
