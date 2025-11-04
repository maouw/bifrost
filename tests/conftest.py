"""Pytest configuration and shared fixtures for bifrost tests."""

import json
from pathlib import Path

import ants
import h5py
import numpy as np
import pytest


@pytest.fixture
def sample_3d_image() -> ants.ANTsImage:
    """Create a simple 3D test image with metadata."""
    shape = (64, 32, 16)
    data = np.random.rand(*shape).astype(np.float32) * 100

    # Add some structure to make it more realistic
    center = tuple(s // 2 for s in shape)
    y, x, z = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask = ((x - center[1]) ** 2 + (y - center[0]) ** 2 + (z - center[2]) ** 2) < (min(shape) // 3) ** 2
    data[mask] = data[mask] * 2  # Brighter center

    image = ants.from_numpy(
        data,
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=np.diag((-1.0, -1.0, 1.0)),
        has_components=False,
    )
    return image


@pytest.fixture
def sample_image_pair(sample_3d_image: ants.ANTsImage) -> tuple[ants.ANTsImage, ants.ANTsImage]:
    """Create a pair of slightly different test images (fixed and moving)."""
    fixed = sample_3d_image

    # Create moving image with slight translation
    moving_data = np.roll(sample_3d_image.numpy(), shift=5, axis=0)
    moving = ants.from_numpy(
        moving_data,
        origin=sample_3d_image.origin,
        spacing=sample_3d_image.spacing,
        direction=sample_3d_image.direction,
        has_components=sample_3d_image.has_components,
    )

    return fixed, moving


@pytest.fixture
def sample_nifti_file(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> Path:
    """Create a temporary NIfTI file."""
    nifti_path = tmp_path / "test_image.nii"
    ants.image_write(sample_3d_image, str(nifti_path))
    return nifti_path


@pytest.fixture
def sample_nifti_pair(tmp_path: Path, sample_image_pair: tuple[ants.ANTsImage, ants.ANTsImage]) -> tuple[Path, Path]:
    """Create a pair of temporary NIfTI files."""
    fixed, moving = sample_image_pair

    fixed_path = tmp_path / "fixed.nii"
    moving_path = tmp_path / "moving.nii"

    ants.image_write(fixed, str(fixed_path))
    ants.image_write(moving, str(moving_path))

    return fixed_path, moving_path


@pytest.fixture
def sample_transform_h5(tmp_path: Path, sample_image_pair: tuple[ants.ANTsImage, ants.ANTsImage]) -> Path:
    """Create a sample transform.h5 file with minimal registration data."""
    fixed, moving = sample_image_pair

    transform_dir = tmp_path / "registration_results"
    transform_dir.mkdir()
    transform_path = transform_dir / "transform.h5"

    with h5py.File(str(transform_path), "w") as h5_handle:
        # Store fixed image metadata
        h5_handle.attrs["fixed.shape"] = fixed.shape
        h5_handle.attrs["fixed.origin"] = fixed.origin
        h5_handle.attrs["fixed.spacing"] = fixed.spacing
        h5_handle.attrs["fixed.direction"] = fixed.direction
        h5_handle.attrs["fixed.has_components"] = fixed.has_components

        # Store registration args
        h5_handle.attrs["args.moving"] = "test_moving.nii"
        h5_handle.attrs["args.fixed"] = "test_fixed.nii"
        h5_handle.attrs["args.downsample_to"] = -1
        h5_handle.attrs["args.moving_clip_limit"] = -1
        h5_handle.attrs["args.clahe_kernel_size"] = 64

        # Store fake affine transform
        h5_handle.create_group("/affine")
        h5_handle.create_dataset("/affine/parameters", data=np.eye(4).flatten()[:12])
        h5_handle.create_dataset("/affine/fixed_parameters", data=np.zeros(3))

    return transform_dir


@pytest.fixture
def mock_weights_file(tmp_path: Path) -> Path:
    """Create a mock SynthMorph weights file with minimal structure."""
    weights_path = tmp_path / "mock_weights.h5"

    with h5py.File(str(weights_path), "w") as h5_handle:
        # Create minimal model config that read_weights_inshape expects

        model_config = {"config": {"inshape": [160, 160, 192]}}
        h5_handle.attrs["model_config"] = json.dumps(model_config)

    return weights_path


@pytest.fixture
def sample_dataset_structure(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> Path:
    """Create a minimal BIFROST dataset structure for Snakemake testing."""
    dataset_dir = tmp_path / "dataset"

    # Create directory structure
    data_dir = dataset_dir / "data"
    templates_dir = dataset_dir / "templates"

    (data_dir / "sample_1" / "channel_1").mkdir(parents=True)
    (data_dir / "sample_2" / "channel_1").mkdir(parents=True)
    templates_dir.mkdir(parents=True)

    # Create sample images
    for sample in ["sample_1", "sample_2"]:
        sample_dir = data_dir / sample

        # Structural image
        structural_path = sample_dir / "structural_image.nii"
        ants.image_write(sample_3d_image, str(structural_path))

        # Dependent image in channel
        dependent_path = sample_dir / "channel_1" / "dependent.nii"
        ants.image_write(sample_3d_image, str(dependent_path))

    # Create FDA template
    fda_path = templates_dir / "FDA.nii"
    ants.image_write(sample_3d_image, str(fda_path))

    return dataset_dir


@pytest.fixture
def label_image(sample_3d_image: ants.ANTsImage) -> ants.ANTsImage:
    """Create a label image (integer labels) for testing."""
    shape = sample_3d_image.shape
    labels = np.zeros(shape, dtype=np.int32)

    # Create some labeled regions
    labels[20:20, 10:20, 1:2] = 1
    labels[30:40, 20:25, 2:7] = 2
    labels[15:25, 25:30, 5:10] = 3

    label_img = ants.from_numpy(
        labels.astype(np.float32),
        origin=sample_3d_image.origin,
        spacing=sample_3d_image.spacing,
        direction=sample_3d_image.direction,
        has_components=sample_3d_image.has_components,
    )

    return label_img
