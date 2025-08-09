"""Module for I/O related methods."""

import hashlib
import logging
import os

import ants
import h5py


def write_affine(h5_handle: h5py.File, name: str, transform: ants.ANTsTransform | str) -> None:
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
    

    h5_handle.create_group(name)
    h5_handle.create_dataset(f"{name}/parameters", data=transform.parameters) # type: ignore
    h5_handle.create_dataset(f"{name}/fixed_parameters", data=transform.fixed_parameters) # type: ignore


def read_affine(h5_handle: h5py.File, name: str, directory: str | None = None) -> ants.ANTsTransform | str:
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

    if directory is None:
        return transform

    transform_path = f"{directory}/aff.mat"

    assert not os.path.exists(transform_path)

    ants.write_transform(transform, transform_path)

    return transform_path


def write_image(h5_handle: h5py.File, name: str, image: ants.ANTsImage | str) -> None:
    """Writes ANTs image to h5.

    Args:
        h5_handle: open file handle - h5py.File
        name: name of dataset to create, absolute path - str
        image: if str interpreted as path to image file - ANTsImage or str
    """
    assert isinstance(h5_handle, h5py.File)
    assert isinstance(image, ants.ANTsImage | str)

    if isinstance(image, str):
        image = ants.image_read(image)

    image_arr = image.numpy() # type: ignore

    h5_handle.create_dataset(
        name,
        data=image_arr,
        chunks=True,
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )

    h5_handle[name].attrs["origin"] = image.origin # type: ignore
    h5_handle[name].attrs["spacing"] = image.spacing # type: ignore
    h5_handle[name].attrs["direction"] = image.direction # type: ignore
    h5_handle[name].attrs["has_components"] = image.has_components # type: ignore


def read_image(h5_handle: h5py.File, name: str, directory: str | None = None) -> ants.ANTsImage | str:
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
    assert isinstance(h5_handle, h5py.File)

    image = ants.from_numpy(
        h5_handle[name][:],
        origin=tuple(h5_handle[name].attrs["origin"]),
        spacing=tuple(h5_handle[name].attrs["spacing"]),
        direction=h5_handle[name].attrs["direction"],
        has_components=h5_handle[name].attrs["has_components"],
    )

    if directory is None:
        return image

    img_path = f"{directory}/img.nii"

    assert not os.path.exists(img_path)

    ants.image_write(image, img_path)

    return img_path


def guarded_ants_image_read(image_path: str) -> ants.ANTsImage:
    """Reads an ANTs image using ants.image_read.

    Raises an exception for multi-channel images.

    Args:
        image_path: path to the image file
    
    Rauses:
        RuntimeError: if the image has multiple channels
    
    Returns:
        image: ANTsImage object
    
    """
    image = ants.image_read(image_path)

    if image.components > 1:
        logger = logging.getLogger(__name__)
        msg = (
            f"{image_path} has multiple channels. "
            "Multi-channel images are not supported by bifrost, "
            "split into per-channel images"
        )
        logger.critical(msg)
        raise RuntimeError(msg)
    return image


def md5sum(filename: str) -> str:
    """Compute the md5sum of a file.

    Args:
        filename: path to file

    Returns:
        md5 hash of the file as a hex string
    """
    file_hash = hashlib.md5()

    with open(filename, "rb") as file_handle:
        chunk = file_handle.read(8192)

        while chunk:
            file_hash.update(chunk)
            chunk = file_handle.read(8192)

    return file_hash.hexdigest()
