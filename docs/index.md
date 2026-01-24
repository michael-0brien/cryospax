# Welcome to cryoSPAX!

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

## Acknowledgements

- CryoSPAX is made possible by the [`teamtomo`](https://teamtomo.org/) ecosystem for open source cryo-EM software (e.g. [`starfile`](https://github.com/teamtomo/starfile)).
