import functools as ft

import equinox as eqx
from jaxtyping import PyTree

from .base_dataset import AbstractDataset
from .relion import RelionParticleDataset, RelionParticleParameterFile


@ft.singledispatch
def get_in_axes(dataset: AbstractDataset) -> PyTree:
    """For a function that accepts an output like `value = dataset[0:10]`,
    this function returns a prefix pytree for input to `eqx.filter_vmap`.

    !!! example
        ```python
        import cryospax as spx

        dataset = RelionParticleDataset(...)

        @eqx.filter_vmap(in_axes=(spx.get_in_axes(dataset), None))
        def fn_vmap(particle_info, args):
            ...

        particle_info = dataset[0:10]
        args = ...  # other arguments to `fn_vmap`
        out = fn_vmap(particle_info, args)
        ```

    **Arguments:**

    - `dataset`:
        A [`cryospax.AbstractDataset`][].

    **Returns:**

    The `in_axes` argument for `equinox.filter_vmap` for the output
    of `dataset[...]`.
    """
    raise NotImplementedError(
        "`cryospax.get_in_axes` not implemented for "
        f"dataset `{dataset.__class__.__name__}`."
    )


@get_in_axes.register(RelionParticleParameterFile)
def _(dataset):
    if dataset.loads_metadata:
        raise AttributeError(
            f"Passed class `dataset = {dataset.__class__.__name__}(...)` "
            "to `cryospax.get_in_axes(dataset)`, but found that "
            "`dataset.loads_metadata = True`. This is not supported "
            "as the output of `dataset[...]` will not be able to pass through "
            "calls to `jax.vmap`."
        )
    return _get_rln_in_axes()


@get_in_axes.register(RelionParticleDataset)
def _(dataset):
    del dataset
    return dict(
        images=eqx.if_array(0),
        parameters=_get_rln_in_axes(),
    )


def _get_rln_in_axes():
    return dict(pose=eqx.if_array(0), transfer_theory=eqx.if_array(0), image_config=None)
