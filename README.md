<h1 align='center'>Cryo-EM single particle analysis in JAX</h1>

[![Continuous Integration](https://github.com/michael-0brien/cryojax-spa/actions/workflows/ci_build.yml/badge.svg)](https://github.com/michael-0brien/cryojax-spa/actions/workflows/ci_build.yml?branch=dev)
[![codecov](https://codecov.io/gh/michael-0brien/cryojax-spa/branch/dev/graph/badge.svg)](https://codecov.io/gh/michael-0brien/cryojax-spa)

# Installation

# Quick example

## Load a RELION STAR file

```python
import cryojax_spa as spa

# Instantiate RELION dataset
parameter_file = spa.RelionParticleParameterFile(path_to_starfile="./path/to/particles.star")
dataset = spa.RelionParticleDataset(parameter_file, path_to_relion_project="./path/to/project/")
# Load first 10 images and particle parameters into `cryojax` classes
particle_info = dataset[0:10]
images, parameters = particle_info["images"], particle_info["parameters"]
```

# Acknowledgements
