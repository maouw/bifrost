from os import PathLike

__all__ = ["Pathable"]

Pathable = str | bytes | PathLike[str] | PathLike[bytes]
"An object which represents a file system path and can be converted to a `pathlib.Path`"
