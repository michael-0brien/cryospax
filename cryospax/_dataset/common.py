import warnings
from collections.abc import Callable
from typing import Literal, TypedDict

import equinox as eqx
import numpy as np
from cryojax.ndimage import FourierConstant, FourierGaussian
from cryojax.simulator import AstigmaticCTF, BasicImageConfig, ContrastTransferTheory


MakeImageConfig = Callable[
    [tuple[int, int], np.ndarray | float, np.ndarray | float], BasicImageConfig
]


def _default_make_image_config(shape, pixel_size, voltage_in_kilovolts):
    """Default implementation for generating an `image_config`
    from parameters and the image shape. Additional options passed
    to `BasicImageConfig` may be desired.
    """
    return eqx.tree_at(
        lambda x: (x.pixel_size, x.voltage_in_kilovolts),
        BasicImageConfig(shape, 1.0, 1.0),
        (pixel_size, voltage_in_kilovolts),
    )


def _validate_mode(mode: str) -> Literal["r", "w"]:
    if mode not in ["r", "w"]:
        raise ValueError(
            f"Passed unsupported `mode = {mode}`. Supported modes are 'r' and 'w'."
        )
    return mode  # type: ignore


def _dict_to_options(d: dict[str, Any]) -> _Options:
    _options_keys = {
        "loads_metadata",
        "loads_envelope",
        "updates_optics_group",
        "make_image_config",
    }
    if not set(d.keys()).issubset(_options_keys):
        raise ValueError(
            "Expected that dictionary `options` passed to "
            "`RelionParticleParameterFile(..., options=...)` "
            f"had a subset of keys {_options_keys}, but found that it "
            f"had keys {set(d.keys())}."
        )
    loads_metadata = d["loads_metadata"] if "loads_metadata" in d else False
    loads_envelope = d["loads_envelope"] if "loads_envelope" in d else False
    updates_optics_group = (
        d["updates_optics_group"] if "updates_optics_group" in d else False
    )
    make_image_config = (
        d["make_image_config"] if "make_image_config" in d else _default_make_image_config
    )
    return _Options(
        loads_metadata=loads_metadata,
        loads_envelope=loads_envelope,
        updates_optics_group=updates_optics_group,
        make_image_config=make_image_config,
    )


def _validate_dataset_index(cls, index, n_rows):
    index_error_msg = lambda idx: (
        f"The index at which the `{cls.__name__}` was accessed was out of bounds! "
        f"The number of rows in the dataset is {n_rows}, but you tried to "
        f"access the index {idx}."
    )
    # ... pandas has bad error messages for its indexing
    if isinstance(index, (int, np.integer)):  # type: ignore
        if index > n_rows - 1:
            raise IndexError(index_error_msg(index))
    elif isinstance(index, slice):
        if index.start is not None and index.start > n_rows - 1:
            raise IndexError(index_error_msg(index.start))
    elif isinstance(index, np.ndarray):
        if index.size == 0:
            raise IndexError(
                "Found that the index passed to the dataset "
                "was an empty numpy array. Please pass a "
                "supported index."
            )
    else:
        raise IndexError(
            f"Indexing with the type {type(index)} is not supported by "
            f"`{cls.__name__}`. Indexing by integers is supported, one-dimensional "
            "fancy indexing is supported, and numpy-array indexing is supported. "
            "For example, like `particle = particle_dataset[0]`, "
            "`particle_stack = particle_dataset[0:5]`, "
            "or `particle_stack = dataset[np.array([1, 4, 3, 2])]`."
        )


def _make_envelope_function(amp, b_factor):
    if b_factor is None and amp is None:
        warnings.warn(
            "`loads_envelope` was set to True, but no envelope parameters were found. "
            "Setting envelope as None. "
            "Make sure your starfile is correctly formatted or set "
            "`loads_envelope=False`."
        )
        return None

    elif b_factor is None and amp is not None:
        return eqx.tree_at(lambda x: x.value, FourierConstant(1.0), amp)
    else:
        if amp is None:
            amp = np.asarray(1.0) if b_factor.ndim == 0 else np.ones_like(b_factor)
        return eqx.tree_at(
            lambda x: (x.amplitude, x.b_factor),
            FourierGaussian(1.0, 1.0),
            (amp, b_factor),
        )


def _make_transfer_theory(defocus, astig, angle, sph, ac, ps, env=None):
    ctf = eqx.tree_at(
        lambda x: (
            x.defocus_in_angstroms,
            x.astigmatism_in_angstroms,
            x.astigmatism_angle,
            x.spherical_aberration_in_mm,
        ),
        AstigmaticCTF(),
        (defocus, astig, angle, sph),
    )
    transfer_theory = ContrastTransferTheory(
        ctf, envelope=env, amplitude_contrast_ratio=0.1, phase_shift=0.0
    )

    return eqx.tree_at(
        lambda x: (x.amplitude_contrast_ratio, x.phase_shift), transfer_theory, (ac, ps)
    )


