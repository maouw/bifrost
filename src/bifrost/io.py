"""Module for I/O related methods."""

import hashlib
import json
from pathlib import Path
from typing import Any

import ants
import h5py
import numpy as np
from ants import ANTsImage, ANTsTransform

from bifrost.types import Pathable


def write_affine(h5_handle: h5py.File, name: str, transform: ANTsTransform | str) -> None:
    """Write ANTs affine transform to h5.

    Args:
        h5_handle: open file handle - h5py.File
        name: name of group to create, absolute path - str
        transform: if str interpreted as path to .mat file - ANTsTransform or str

    Returns:
        None

    """
    if isinstance(transform, str):
        transform = ants.read_transform(transform)
    assert isinstance(transform, ANTsTransform), "transform must be an ANTsTransform or a path to a .mat file"
    h5_handle.create_group(name)
    h5_handle.create_dataset(f"{name}/parameters", data=transform.parameters)
    h5_handle.create_dataset(f"{name}/fixed_parameters", data=transform.fixed_parameters)


def read_affine(h5_handle: h5py.File, name: str, directory: Pathable = None) -> ANTsTransform | str:
    """Read ANTs affine transform from h5.

    If directory is not None, writes to a file and returns absolute path
    This allows use with ants.apply_transforms which demands files.


    Args:
        h5_handle: open file handle - h5py.File
        name: name of group, absolute path - str
        directory: absolute path to (presumably temporary) directory to write result to

    Returns:
        transform: path or transform object
    """
    transform = ants.create_ants_transform()
    transform.set_parameters(h5_handle[f"{name}/parameters"][:])
    transform.set_fixed_parameters(h5_handle[f"{name}/fixed_parameters"][:])

    transform = ants.create_ants_transform()
    assert isinstance(transform, ANTsTransform)
    assert isinstance(h5_handle, h5py.File)
    transform.set_parameters(h5_handle[f"{name}/parameters"][:])
    transform.set_fixed_parameters(h5_handle[f"{name}/fixed_parameters"][:])

    if directory is None:
        return transform

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    transform_path = directory / "aff.mat"
    if transform_path.exists():
        raise FileExistsError(
            f"Transform file {transform_path} already exists. Please choose a different directory or remove the existing file."
        )

    ants.write_transform(transform, str(transform_path))

    return str(transform_path)


def write_image(h5_handle: h5py.File, name: str | Path, image: ANTsImage | str | Path) -> None:
    """Writes ANTs image to h5.

    Args:
        h5_handle: open file handle - h5py.File
        name: name of dataset to create, absolute path - str
        image: if str interpreted as path to image file - ANTsImage or str
    """
    name = name if isinstance(name, str) else str(name)
    image = image if isinstance(image, ANTsImage) else ants.image_read(str(image))
    image_arr = image.numpy()
    h5_handle.create_dataset(
        name,
        data=image_arr,
        chunks=True,
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    h5_handle[name].attrs["origin"] = image.origin
    h5_handle[name].attrs["spacing"] = image.spacing
    h5_handle[name].attrs["direction"] = image.direction
    h5_handle[name].attrs["has_components"] = image.has_components


def read_image(h5_handle: h5py.File, name: Pathable, directory: Pathable | None = None) -> ANTsImage | Path:
    """Reads ANTs image from h5.

    If directory is not None, writes to a file and returns absolute path
    This allows use with ants.apply_transforms which demands files.


    Args:
        h5_handle: open file handle - h5py.File
        name: name of dataset, absolute path - str
        directory: absolute path to (presumably temporary) directory to write result to

    Returns:
        image - ants.ANTs.Image
    """
    name = str(name)
    if directory is not None:
        directory = Path(directory)
    dset = h5_handle[str(name)]  # type: ignore
    image = ants.from_numpy(
        dset[:],
        origin=tuple(dset.attrs["origin"]),
        spacing=tuple(dset.attrs["spacing"]),
        direction=dset.attrs["direction"],
        has_components=dset.attrs["has_components"],
    )
    if directory is None:
        return image
    img_path = Path(directory, "image.nii")
    if img_path.exists():
        raise FileExistsError(
            f"Image file {img_path} already exists. Please choose a different directory or remove the existing file."
        )
    ants.image_write(image, str(img_path))
    return Path(img_path)


def check_ants_header(filename: Pathable) -> dict[str, Any]:
    hdr = ants.core.ants_image_io.image_header_info(str(filename))
    if not hdr:
        raise ValueError(f"Image file {filename} does not have an ANTsImage header")
    if int(hdr.get("nComponents", 1)) > 1:
        raise ValueError("Multi-channel images not supported. Please plit into per-channel images.")
    return hdr


def guarded_ants_image_read(image_path: Pathable) -> ANTsImage:
    """Reads an ANTs image using ants.image_read.

    Raises an exception for multi-channel images.

    Args:
        image_path: path to the image file

    Rauses:
        RuntimeError: if the image has multiple channels

    Returns:
        image: ANTsImage object

    """
    check_ants_header(image_path)
    return ants.image_read(str(image_path))


def md5sum(filename: Pathable, chunk_size: int = 8192) -> str:
    """Compute the md5sum of a file.

    Args:
        filename: path to file
        chunk_size: size of chunks to read from file

    Returns:
        md5 hash of the file as a hex string
    """
    file_hash = hashlib.md5()
    with Path(filename).open("rb") as f:
        while chunk := f.read(chunk_size):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def read_weights_inshape(path: Pathable) -> tuple[int, int, int]:
    """Reads the 'inshape' configuration from a model weights HDF5 file.

    Args:
        path: Path to the weights file.

    Returns:
        A tuple of three integers representing the input shape.

    Raises:
        KeyError: If the required 'model_config.config.inshape' attribute is missing.
        ValueError: If the JSON cannot be decoded or the inshape cannot be converted to integers or is not a 3-tuple.
    """
    with h5py.File(str(path), "r") as h5_handle:
        try:
            inshape_obj: list[int | float] = json.loads(str(h5_handle.attrs["model_config"]))["config"]["inshape"]
        except KeyError as e:
            raise KeyError("The weights file does not contain a 'model_config.config.inshape' attribute: {e}") from e
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError("Failed to decode 'model_config' JSON") from e
        try:
            inshape = np.asarray(inshape_obj).astype(dtype=int, casting="safe")
        except ValueError as e:
            raise ValueError("Failed to convert 'inshape' to integer type") from e
        return tuple(inshape)
