"""Functionality in `cryospax` for datasets."""

import abc
import pathlib
from copy import deepcopy
from typing import Generic, Literal, TypeVar
from typing_extensions import Self

import numpy as np
from cryojax.jax_util import NDArrayLike
from jaxtyping import Float, Int, PyTree


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")


class AbstractDataset(abc.ABC, Generic[T]):
    """An abstraction of a dataset in `cryospax`. To create an
    `AbstractDataset`, implement its `__init__`, `__getitem__`, and
    `__len__` methods.

    This follows the
    [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
    API and can easily be wrapped into a pytorch `Dataset` with something like the
    following pattern:

    ```python
    import numpy as np
    from torch.utils.data import Dataset

    class CustomTorchDataset(Dataset):

        def __init__(cryojax_dataset: AbstractDataset):
            self.cryojax_dataset = cryojax_dataset

        def __getitem___(self, index) -> dict[str, np.ndarray]:
            particle_info = self.cryojax_dataset[index]
            return dict(index=index, images=np.asarray(particle_info["images"]))

        def __len__(self) -> int:
            return len(self.cryojax_dataset)
    ```

    JAX also includes packages for dataloaders, such as
    [`jax-dataloaders`](https://github.com/BirkhoffG/jax-dataloader/tree/main) and
    [`grain`](https://github.com/google/grain).

    !!! question "How do I implement an `AbstractDataset`?"

        Implementing an `AbstractDataset` is not like implementing
        classes in `cryojax`, which are `equinox.Module`s.
        An `equinox.Module` is just a pytree, so it can be safely
        passed to `jax` transformations. However, an `AbstractDataset`
        can *not* be passed to `jax` transformations. It is
        a normal python class, not a pytree.

    """

    @abc.abstractmethod
    def __getitem__(self, index) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class AbstractParticleParameterFile(AbstractDataset[T1], Generic[T1, T2]):
    """Base class for a particle parameter file."""

    @abc.abstractmethod
    def __setitem__(self, index, value: T2):
        raise NotImplementedError

    @abc.abstractmethod
    def append(self, value: T2):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def path_to_output(self) -> pathlib.Path:
        raise NotImplementedError

    @path_to_output.setter
    @abc.abstractmethod
    def path_to_output(self, value: str | pathlib.Path):
        raise NotImplementedError

    @property
    def mode(self) -> Literal["r", "w"]:
        raise NotImplementedError

    def copy(self) -> Self:
        """Make a deep copy of the `parameter_file`. This is
        often useful when modifying via `parameter_file[...] = value`
        syntax and it is necessary to maintain a copy of the original
        data.
        """
        return deepcopy(self)


class AbstractParticleDataset(AbstractDataset[T1], Generic[T1, T2]):
    """Base class for a particle dataset."""

    @property
    @abc.abstractmethod
    def parameter_file(self) -> AbstractParticleParameterFile:
        raise NotImplementedError

    @abc.abstractmethod
    def __setitem__(self, index, value: T2):
        raise NotImplementedError

    @abc.abstractmethod
    def append(self, value: T2):
        raise NotImplementedError

    @abc.abstractmethod
    def write_images(
        self,
        index_array: Int[np.ndarray, " _"],
        images: Float[NDArrayLike, "... _ _"],
        parameters: PyTree | None = None,
    ):
        raise NotImplementedError

    @property
    def mode(self) -> Literal["r", "w"]:
        raise NotImplementedError
