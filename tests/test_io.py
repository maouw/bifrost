"""Tests for bifrost I/O module."""

import json
from pathlib import Path

import ants
import h5py
import numpy as np
import pytest

from bifrost.io import (
    guarded_ants_image_read,
    md5sum,
    read_affine,
    read_image,
    read_weights_inshape,
    write_affine,
    write_image,
)

"""Tests for image read/write functions."""


def test_write_and_read_image(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test writing and reading ANTs image to/from HDF5."""
    h5_path = tmp_path / "test.h5"

    with h5py.File(str(h5_path), "w") as h5_handle:
        write_image(h5_handle, "/test_image", sample_3d_image)

    with h5py.File(str(h5_path), "r") as h5_handle:
        loaded_image = read_image(h5_handle, "/test_image")
        assert isinstance(loaded_image, ants.ANTsImage)

    # Verify data and metadata are preserved
    np.testing.assert_array_almost_equal(sample_3d_image.numpy(), loaded_image.numpy())
    assert sample_3d_image.origin == loaded_image.origin
    assert sample_3d_image.spacing == loaded_image.spacing
    np.testing.assert_array_equal(sample_3d_image.direction, loaded_image.direction)
    assert sample_3d_image.has_components == loaded_image.has_components


def test_write_image_from_path(tmp_path: Path, sample_nifti_file: Path) -> None:
    """Test writing image to HDF5 from file path."""
    h5_path = tmp_path / "test.h5"

    with h5py.File(str(h5_path), "w") as h5_handle:
        write_image(h5_handle, "/test_image", str(sample_nifti_file))

    with h5py.File(str(h5_path), "r") as h5_handle:
        assert "/test_image" in h5_handle
        assert h5_handle["/test_image"].shape is not None


def test_read_image_to_file(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test reading image from HDF5 and writing to file."""
    h5_path = tmp_path / "test.h5"

    with h5py.File(str(h5_path), "w") as h5_handle:
        write_image(h5_handle, "/test_image", sample_3d_image)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with h5py.File(str(h5_path), "r") as h5_handle:
        image_path = read_image(h5_handle, "/test_image", directory=str(output_dir))

    assert isinstance(image_path, Path)
    assert image_path.exists()
    loaded_image = ants.image_read(str(image_path))
    np.testing.assert_array_almost_equal(sample_3d_image.numpy(), loaded_image.numpy())


def test_read_image_file_exists_error(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test error when trying to write image to existing file."""
    h5_path = tmp_path / "test.h5"

    with h5py.File(str(h5_path), "w") as h5_handle:
        write_image(h5_handle, "/test_image", sample_3d_image)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "image.nii").touch()  # Create existing file

    with h5py.File(str(h5_path), "r") as h5_handle, pytest.raises(FileExistsError):
        read_image(h5_handle, "/test_image", directory=str(output_dir))


def test_guarded_ants_image_read_single_channel(sample_nifti_file: Path) -> None:
    """Test reading single-channel image."""
    image = guarded_ants_image_read(str(sample_nifti_file))
    assert image is not None
    assert image.components == 1


def test_guarded_ants_image_read_multichannel_error(tmp_path: Path) -> None:
    """Test error when reading multi-channel image."""
    # Create multi-channel image
    data = np.random.rand(50, 60, 70, 3).astype(np.float32)
    multichannel_img = ants.from_numpy(data, has_components=True)
    multichannel_path = tmp_path / "multichannel.nii"
    ants.image_write(multichannel_img, str(multichannel_path))

    with pytest.raises(ValueError):
        guarded_ants_image_read(str(multichannel_path))


"""Tests for affine transform read/write functions."""


def test_write_and_read_affine(tmp_path: Path, sample_image_pair: tuple[Path, Path]) -> None:
    """Test writing and reading ANTs affine transform to/from HDF5."""
    fixed, moving = sample_image_pair

    # Create affine transform
    registration = ants.registration(fixed, moving, type_of_transform="Affine", outprefix=str(tmp_path / "affine_"))
    affine_transform = ants.read_transform(registration["fwdtransforms"][0])

    h5_path = tmp_path / "test.h5"

    with h5py.File(str(h5_path), "w") as h5_handle:
        write_affine(h5_handle, "/affine", affine_transform)

    with h5py.File(str(h5_path), "r") as h5_handle:
        loaded_transform = read_affine(h5_handle, "/affine")

    # Verify transform parameters are preserved
    assert isinstance(loaded_transform, ants.ANTsTransform)
    np.testing.assert_array_almost_equal(affine_transform.parameters, loaded_transform.parameters)
    np.testing.assert_array_almost_equal(affine_transform.fixed_parameters, loaded_transform.fixed_parameters)


def test_write_affine_from_path(tmp_path: Path, sample_image_pair: tuple[Path, Path]) -> None:
    """Test writing affine transform to HDF5 from file path."""
    fixed, moving = sample_image_pair

    registration = ants.registration(fixed, moving, type_of_transform="Affine", outprefix=str(tmp_path / "affine_"))
    mat_path = registration["fwdtransforms"][0]

    h5_path = tmp_path / "test.h5"

    with h5py.File(str(h5_path), "w") as h5_handle:
        write_affine(h5_handle, "/affine", mat_path)

    with h5py.File(str(h5_path), "r") as h5_handle:
        assert "/affine" in h5_handle
        assert "/affine/parameters" in h5_handle
        assert "/affine/fixed_parameters" in h5_handle


def test_read_affine_to_file(tmp_path: Path, sample_image_pair: tuple[Path, Path]) -> None:
    """Test reading affine from HDF5 and writing to file."""
    fixed, moving = sample_image_pair

    registration = ants.registration(fixed, moving, type_of_transform="Affine", outprefix=str(tmp_path / "affine_"))
    affine_transform = ants.read_transform(registration["fwdtransforms"][0])

    h5_path = tmp_path / "test.h5"

    with h5py.File(str(h5_path), "w") as h5_handle:
        write_affine(h5_handle, "/affine", affine_transform)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with h5py.File(str(h5_path), "r") as h5_handle:
        affine_path = read_affine(h5_handle, "/affine", directory=str(output_dir))

    assert isinstance(affine_path, str)
    assert Path(affine_path).exists()
    assert Path(affine_path).name == "aff.mat"


def test_read_affine_file_exists_error(tmp_path: Path, sample_image_pair: tuple[Path, Path]) -> None:
    """Test error when trying to write affine to existing file."""
    fixed, moving = sample_image_pair

    registration = ants.registration(fixed, moving, type_of_transform="Affine", outprefix=str(tmp_path / "affine_"))
    affine_transform = ants.read_transform(registration["fwdtransforms"][0])

    h5_path = tmp_path / "test.h5"

    with h5py.File(str(h5_path), "w") as h5_handle:
        write_affine(h5_handle, "/affine", affine_transform)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "aff.mat").touch()  # Create existing file

    with h5py.File(str(h5_path), "r") as h5_handle, pytest.raises(FileExistsError):
        read_affine(h5_handle, "/affine", directory=str(output_dir))


def test_read_affine_directory_not_found(tmp_path: Path, sample_image_pair: tuple[Path, Path]) -> None:
    """Test error when output directory doesn't exist."""
    fixed, moving = sample_image_pair

    registration = ants.registration(fixed, moving, type_of_transform="Affine", outprefix=str(tmp_path / "affine_"))
    affine_transform = ants.read_transform(registration["fwdtransforms"][0])

    h5_path = tmp_path / "test.h5"

    with h5py.File(str(h5_path), "w") as h5_handle:
        write_affine(h5_handle, "/affine", affine_transform)

    with h5py.File(str(h5_path), "r") as h5_handle, pytest.raises(FileNotFoundError):
        read_affine(h5_handle, "/affine", directory=str(tmp_path / "nonexistent"))


"""Tests for utility I/O functions."""


def test_md5sum(tmp_path: Path) -> None:
    """Test MD5 checksum calculation."""
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, World!"
    test_file.write_bytes(test_content)

    checksum = md5sum(str(test_file))

    assert isinstance(checksum, str)
    assert len(checksum) == 32  # MD5 produces 32 hex characters
    # Verify consistency
    assert md5sum(str(test_file)) == checksum


def test_md5sum_different_files(tmp_path: Path) -> None:
    """Test that different files have different checksums."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"

    file1.write_bytes(b"Content A")
    file2.write_bytes(b"Content B")

    checksum1 = md5sum(str(file1))
    checksum2 = md5sum(str(file2))

    assert checksum1 != checksum2


def test_read_weights_inshape(mock_weights_file: Path) -> None:
    """Test reading inshape from weights file."""
    inshape = read_weights_inshape(str(mock_weights_file))

    assert isinstance(inshape, tuple)
    assert len(inshape) == 3
    assert inshape == (160, 160, 192)


def test_read_weights_inshape_missing_attr(tmp_path: Path) -> None:
    """Test error when model_config attribute is missing."""
    weights_path = tmp_path / "invalid_weights.h5"

    with h5py.File(str(weights_path), "w") as h5_handle:
        pass  # Create empty file

    with pytest.raises(KeyError):
        read_weights_inshape(str(weights_path))


def test_read_weights_inshape_invalid_json(tmp_path: Path) -> None:
    """Test error when model_config is not valid JSON."""
    weights_path = tmp_path / "invalid_weights.h5"

    with h5py.File(str(weights_path), "w") as h5_handle:
        h5_handle.attrs["model_config"] = "not valid json"

    with pytest.raises(ValueError):
        read_weights_inshape(str(weights_path))


def test_read_weights_inshape_wrong_type(tmp_path: Path) -> None:
    """Test error when inshape cannot be converted to integers."""

    weights_path = tmp_path / "invalid_weights.h5"

    with h5py.File(str(weights_path), "w") as h5_handle:
        model_config = {"config": {"inshape": ["not", "a", "number"]}}
        h5_handle.attrs["model_config"] = json.dumps(model_config)

    with pytest.raises(TypeError):
        read_weights_inshape(str(weights_path))
