"""Module for I/O related methods."""

import hashlib
import json
import logging
from pathlib import Path

import ants
import h5py
import numpy as np


def write_affine(h5_handle: h5py.File, name: str, transform: ants.ANTsTransform | str | Path) -> None:
    """Write ANTs affine transform to h5.

    Args:
        h5_handle: open file handle - h5py.File
        name: name of group to create, absolute path - str
        transform: if str interpreted as path to .mat file - ANTsTransform or str

    Returns:
        None

    """
    if isinstance(transform, (str | Path)):
        transform = ants.read_transform(str(transform))
    assert isinstance(transform, ants.ANTsTransform), "transform must be an ANTsTransform or a path to a .mat file"
    name = str(name)
    h5_handle.create_group(name)
    h5_handle.create_dataset(f"{name}/parameters", data=transform.parameters)
    h5_handle.create_dataset(f"{name}/fixed_parameters", data=transform.fixed_parameters)


def read_affine(h5_handle: h5py.File, name: str | Path, directory: str | Path | None = None) -> ants.ANTsTransform | str:
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
    assert isinstance(transform, ants.ANTsTransform)
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
        raise FileExistsError(f"Transform file {transform_path} already exists. Please choose a different directory or remove the existing file.")

    ants.write_transform(transform, str(transform_path))

    return str(transform_path)


def write_image(h5_handle: h5py.File, name: str | Path, image: ants.ANTsImage | str | Path) -> None:
    """Writes ANTs image to h5.

    Args:
        h5_handle: open file handle - h5py.File
        name: name of dataset to create, absolute path - str
        image: if str interpreted as path to image file - ANTsImage or str
    """
    name = name if isinstance(name, str) else str(name)
    image = image if isinstance(image, ants.ANTsImage) else ants.image_read(image)
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


def read_image(h5_handle: h5py.File, name: str | Path, directory: str | Path | None = None) -> ants.ANTsImage | Path:
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
    img_path = Path(directory) / "image.nii"
    if img_path.exists():
        raise FileExistsError(f"Image file {img_path} already exists. Please choose a different directory or remove the existing file.")
    ants.image_write(image, str(img_path))
    return img_path


def guarded_ants_image_read(image_path: str | Path) -> ants.ANTsImage:
    """Reads an ANTs image using ants.image_read.

    Raises an exception for multi-channel images.

    Args:
        image_path: path to the image file

    Rauses:
        RuntimeError: if the image has multiple channels

    Returns:
        image: ANTsImage object

    """
    image = ants.image_read(str(image_path))
    if image.components > 1:
        logger = logging.getLogger(__name__)
        msg = f"{image_path} has multiple channels. Multi-channel images are not supported by bifrost, split into per-channel images"
        logger.critical(msg)
        raise RuntimeError(msg)
    return image


def md5sum(filename: str | Path, chunk_size: int = 8192) -> str:
    """Compute the md5sum of a file.

    Args:
        filename: path to file
        chunk_size: size of chunks to read from file

    Returns:
        md5 hash of the file as a hex string
    """
    file_hash = hashlib.md5()
    with open(filename, "rb") as f:
        while chunk := f.read(chunk_size):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def read_weights_inshape(path: str | Path) -> tuple[int, int, int]:
    """Reads the 'inshape' configuration from a model weights HDF5 file.

    Args:
        path: Path to the weights file.

    Returns:
        A tuple of three integers representing the input shape.

    Raises:
        KeyError: If the required 'model_config.config.inshape' attribute is missing.
        ValueError: If the JSON cannot be decoded or the inshape cannot be converted to integers or is not a 3-tuple.
    """
    with h5py.File(str(path), "rb") as h5_handle:
        try:
            inshape_obj: list[int | float] = json.loads(str(h5_handle.attrs["model_config"]))["config"]["inshape"]
        except KeyError as e:
            raise KeyError("The weights file does not contain a 'model_config.config.inshape' attribute: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError("Failed to decode 'model_config' JSON") from e
        try:
            inshape = np.asarray(inshape_obj).astype(dtype=int, casting="safe")
        except ValueError as e:
            raise ValueError("Failed to convert 'inshape' to integer type") from e
        return tuple(inshape)
