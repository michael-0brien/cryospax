import functools as ft

import equinox as eqx
from jaxtyping import PyTree

from .base_dataset import AbstractDataset
from .relion import RelionParticleDataset, RelionParticleParameterFile


@ft.singledispatch
def get_in_axes(dataset: AbstractDataset) -> PyTree:
    """For a function that accepts an output like `value = dataset[0:10]`,
    this function returns a prefix pytree for input to
    [`equinox.filter_vmap`](https://docs.kidger.site/equinox/api/transformations/#vectorisation-and-parallelisation).

    !!! example
        ```python
        import cryospax as spx

        dataset = spx.RelionParticleDataset(...)

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
    """  # noqa: E501
    raise NotImplementedError(
        "`cryospax.get_in_axes` is not implemented for "
        f"dataset of type `{dataset.__class__.__name__}`."
    )


_get_metadata_error_msg = lambda _dataset, _where_string: (
    f"Passed class `dataset = {_dataset.__class__.__name__}(...)` "
    "to `cryospax.get_in_axes(dataset)`, but found that "
    f"`{_where_string} = True`. This is not supported because "
    "the output of `dataset[...]` will not be able to pass through "
    "calls to `equinox.filter_vmap`."
)


@get_in_axes.register(RelionParticleParameterFile)
def _(dataset):
    if dataset.loads_metadata:
        raise AttributeError(_get_metadata_error_msg(dataset, "dataset.loads_metadata"))
    return _get_rln_parameters_in_axes()


@get_in_axes.register(RelionParticleDataset)
def _(dataset):
    if dataset.parameter_file.loads_metadata:
        raise AttributeError(
            _get_metadata_error_msg(dataset, "dataset.parameter_file.loads_metadata")
        )
    if dataset.only_images:
        return dict(images=eqx.if_array(0))
    else:
        return dict(
            images=eqx.if_array(0),
            parameters=_get_rln_parameters_in_axes(),
        )


def _get_rln_parameters_in_axes():
    return dict(pose=eqx.if_array(0), transfer_theory=eqx.if_array(0), image_config=None)
