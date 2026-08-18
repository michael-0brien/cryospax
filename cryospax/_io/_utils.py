import pathlib
from typing import Literal


def _validate_filename(
    filename: str | pathlib.Path, mode: Literal["r", "w"], suffix: Literal["star", "cs"]
):
    suffixes = pathlib.Path(filename).suffixes
    if not (len(suffixes) == 1 and suffixes[0] == f".{suffix}"):
        raise OSError(
            f"Tried to {('write' if mode == 'w' else 'read')} {suffix.upper()} file, "
            f"but the filename does not include a '.{suffix}' "
            f"suffix. Got filename '{filename}'."
        )
