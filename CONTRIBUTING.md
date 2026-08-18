# Contributor Guide

Contributions to this repository are welcome.

## What features belong in `cryospax`?

TODO

## Making a feature request

To submit a feature request, open an a thread on the [issues](https://github.com/michael-0brien/cryospax/issues) page. After discussing the contribution, please implement a draft of it in your local fork of `cryospax`. Then, open a [pull request](https://github.com/michael-0brien/cryospax/pulls).

## Reporting a bug

Make bug reports on the [issues](https://github.com/michael-0brien/cryospax/issues) page. Please provide a test case, and/or steps to reproduce the issue. In particular, consider including a [minimal, reproducible example](https://stackoverflow.com/help/minimal-reproducible-example).

## How to contribute

Let's say you are submitted a bug fix or a feature request to `cryospax`. To contribute, first fork the library on github. Then clone and install the library with dependencies for development and testing:

```
git clone https://github.com/your-username-here/cryospax.git
cd cryospax
git checkout dev
python -m pip install -e '.[dev, tests]'
```

Next, install the pre-commit hooks:

```
pre-commit install
```

This uses `ruff` to format and lint the code. Now, you can push changes to your local fork.

### Running tests

After making changes, make sure that the tests pass. In the `cryospax` base directory, install testing dependencies and run

```
python -m pytest
```

### Submitting changes

If the tests look okay, open a [pull request](https://github.com/michael-0brien/cryospax/pulls) from your fork the `dev` branch. The developers can review your PR and request changes / add further tweaks if necessary.

### Optional: build documentation

For a given PR it may also be necessary to build the `cryospax` documentation or run jupyter notebook examples. The documentation is easily built using [`mkdocs`](https://www.mkdocs.org/getting-started/#getting-started-with-mkdocs). To make sure the docs build, run the following:

```
python -m pip install -e '.[docs]'
mkdocs build
```

You can also run `mkdocs serve` and follow the instructions in your terminal to inspect the webpage on your local server.

To run the notebooks in the documentation, it may be necessary to pull large-ish files from [git LFS](https://git-lfs.com/).

```
sudo apt-get install git-lfs  # If using macOS, `brew install git-lfs`
git lfs install; git lfs pull
```
