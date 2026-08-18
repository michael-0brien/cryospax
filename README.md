<h1 align='center'>Single particle analysis tools for cryoJAX</h1>

[![Continuous Integration](https://github.com/michael-0brien/cryospax/actions/workflows/ci_build.yml/badge.svg)](https://github.com/michael-0brien/cryospax/actions/workflows/ci_build.yml?branch=dev)
[![codecov](https://codecov.io/gh/michael-0brien/cryospax/graph/badge.svg?token=BAYZQFY5FH)](https://codecov.io/gh/michael-0brien/cryospax)

CryoSPAX is a *small* library built to support [cryoJAX](https://github.com/michael-0brien/cryojax) projects that implement single-particle analysis (SPA) at scale. It seeks to simplify new cryo-EM research, rather than providing end-to-end single particle analysis workflows.

## Installation

If you are not installing JAX with GPU or TPU support, installing `cryospax` in a fresh [`uv`](https://docs.astral.sh/uv/pip/environments/#creating-a-virtual-environment) environment is as simple as

```bash
uv venv --python=3.11 ~/path/to/venv/
source ~/path/to/venv/bin/activate
uv pip install cryospax
```

If you are installing JAX with GPU or TPU support, first follow the [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html#installation). It may also be necessary to directly follow the [`cryojax` installation instructions](https://michael-0brien.github.io/cryojax/#installation) for installs with advanced cryoJAX features.

To install `cryospax` in development mode, run

```bash
git clone https://github.com/michael-0brien/cryospax
cd cryospax
git checkout dev
uv pip install -e '.[dev,tests]'
uv run pre-commit install
```

## Reading CryoSPARC files

CryoSPARC particle metadata is stored in `.cs` files, which `cryospax` reads by converting them to a RELION STAR file with the `csparc2spax` program

```bash
cryospax csparc2spax particles.cs particles.star --passthrough passthrough_particles.cs
```

The first two arguments are the CryoSPARC `.cs` file to read and the STAR file to write. The remaining arguments are optional:

- `--passthrough`, the path to a CryoSPARC passthrough particles `.cs` file, whose fields are merged into those of the input file on the particle `'uid'`. CryoSPARC often stores the CTF and optics parameters here rather than in the main file, in which case this argument is required to write a STAR file that can be read back.
- `--overwrite`, to replace an existing output file.

The result is read like any other STAR file

```python
from cryospax import RelionParticleDataset, RelionParticleParameterFile

parameter_file = RelionParticleParameterFile("particles.star")
dataset = RelionParticleDataset(parameter_file, "/path/to/cryosparc/project")
```

Image paths are taken from the CryoSPARC `'blob/path'` field, so the project directory passed to the dataset is the *CryoSPARC* project directory.

> [!WARNING]
> This is **not** a general purpose converter from CryoSPARC to RELION. Only the CryoSPARC fields that `cryospax` interprets are written out, i.e. those needed to build the image configuration, the CTF, and the pose of each particle. Every other field is dropped, so the result is not a substitute for the original file and is not guaranteed to be understood by RELION or by other programs that expect a complete STAR file. For a general purpose conversion, use [`pyem`](https://github.com/asarnow/pyem).

## Acknowledgements

- CryoSPAX is made possible by the [`teamtomo`](https://teamtomo.org/) ecosystem for open source cryo-EM software (e.g. [`starfile`](https://github.com/teamtomo/starfile)).
- The CryoSPARC file support in CryoSPAX is based on [`pyem`](https://github.com/asarnow/pyem) by Daniel Asarnow and [`xmipp_metadata`](https://github.com/DavidHerreros/xmipp_metadata) by David Herreros.
