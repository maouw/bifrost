"""Tests for bifrost transform command."""

import argparse
from pathlib import Path

import ants
import h5py
import numpy as np
import pytest

from bifrost.cli.apply_transform import transform


class TestTransformCommand:
    """Tests for the transform command functionality."""

    def test_transform_basic_workflow(self, tmp_path: Path, sample_transform_h5: Path, sample_nifti_file: Path) -> None:
        """Test basic transform application workflow."""
        image_path = sample_nifti_file

        args = argparse.Namespace(
            alignment_path=str(sample_transform_h5),
            image_path=str(image_path),
            label_image=False,
            apply_preprocessing=False,
            result_name=None,
            log=None,
            verbose=True,
        )

        transform(args)

        # Check that transformed image was created
        result_name = f"{image_path.stem}_transformed.nii"
        result_path = sample_transform_h5 / result_name

        assert result_path.exists()

        # Verify the output is a valid image
        transformed = ants.image_read(str(result_path))
        assert transformed is not None
        assert transformed.shape is not None
        result_path.unlink()

    def test_transform_with_preprocessing(self, tmp_path: Path, sample_transform_h5: Path, sample_nifti_file: Path):
        """Test transform with preprocessing applied."""
        # Update the transform.h5 to include preprocessing parameters
        with h5py.File(str(sample_transform_h5 / "transform.h5"), "r+") as h5_handle:
            h5_handle.attrs["args.moving_clip_limit"] = 0.03

        args = argparse.Namespace(
            alignment_path=str(sample_transform_h5),
            image_path=str(sample_nifti_file),
            label_image=False,
            apply_preprocessing=True,
            result_name=None,
            log=None,
            verbose=True,
        )

        transform(args)

        result_name = f"{sample_nifti_file.stem}_transformed.nii"
        result_path = sample_transform_h5 / result_name
        assert result_path.exists()
        result_path.unlink()

    def test_transform_label_image(self, tmp_path: Path, sample_transform_h5: Path, label_image):
        """Test transform with label (integer) image."""
        label_path = tmp_path / "labels.nii"
        ants.image_write(label_image, str(label_path))

        args = argparse.Namespace(
            alignment_path=str(sample_transform_h5),
            image_path=str(label_path),
            label_image=True,
            apply_preprocessing=False,
            result_name=None,
            log=None,
            verbose=True,
        )

        transform(args)

        result_name = f"{label_path.stem}_transformed.nii"
        result_path = sample_transform_h5 / result_name

        assert result_path.exists()

        # Verify output is still integer-like (nearest neighbor interpolation)
        transformed = ants.image_read(str(result_path))
        # Labels should be preserved (though some smoothing may occur at boundaries)
        unique_labels = np.unique(transformed.numpy())
        assert len(unique_labels) <= len(np.unique(label_image.numpy())) + 5  # Allow some interpolation artifacts
        result_path.unlink()

    def test_transform_custom_result_name(self, tmp_path: Path, sample_transform_h5: Path, sample_nifti_file: Path):
        """Test transform with custom result name."""
        custom_name = "my_custom_result.nii"

        args = argparse.Namespace(
            alignment_path=str(sample_transform_h5),
            image_path=str(sample_nifti_file),
            label_image=False,
            apply_preprocessing=False,
            result_name=custom_name,
            log=None,
            verbose=True,
        )

        transform(args)

        result_path = sample_transform_h5 / custom_name
        assert result_path.exists()
        result_path.unlink()

    def test_transform_custom_result_path(self, tmp_path: Path, sample_transform_h5: Path, sample_nifti_file: Path):
        """Test transform with custom absolute result path."""
        custom_path = tmp_path / "custom_output" / "result.nii"

        args = argparse.Namespace(
            alignment_path=str(sample_transform_h5),
            image_path=str(sample_nifti_file),
            label_image=False,
            apply_preprocessing=False,
            result_name=str(custom_path),
            log=None,
            verbose=True,
        )

        transform(args)

        assert custom_path.exists()
        assert custom_path.parent.exists()
        custom_path.unlink()

    def test_transform_with_syn(self, tmp_path: Path, sample_transform_h5: Path, sample_nifti_file, sample_image_pair):
        """Test transform with SyN registration data."""
        fixed, moving = sample_image_pair

        # Add SyN transform to the h5 file
        with h5py.File(str(sample_transform_h5 / "transform.h5"), "r+") as h5_handle:
            h5_handle.create_group("/syn")

            # Store affine
            h5_handle.create_dataset("/syn/affine/parameters", data=np.eye(3).flatten())
            h5_handle.create_dataset("/syn/affine/fixed_parameters", data=np.zeros(3))

            # Store forward warp (just a dummy displacement field)
            warp = np.random.randn(*fixed.shape, 3).astype(np.float32) * 0.1
            h5_handle.create_dataset("/syn/forward_warp", data=warp, chunks=True, compression="gzip", compression_opts=9)
            h5_handle["/syn/forward_warp"].attrs["origin"] = fixed.origin
            h5_handle["/syn/forward_warp"].attrs["spacing"] = fixed.spacing
            h5_handle["/syn/forward_warp"].attrs["direction"] = fixed.direction
            h5_handle["/syn/forward_warp"].attrs["has_components"] = True

        args = argparse.Namespace(
            alignment_path=str(sample_transform_h5),
            image_path=str(sample_nifti_file),
            label_image=False,
            apply_preprocessing=False,
            result_name=None,
            log=None,
            verbose=True,
        )

        transform(args)

        result_name = f"{sample_nifti_file.stem}_transformed.nii"
        result_path = sample_transform_h5 / result_name
        assert result_path.exists()
        result_path.unlink()

    def test_transform_custom_log(self, tmp_path: Path, sample_transform_h5: Path, sample_nifti_file: Path):
        """Test transform with custom log file."""
        log_path = tmp_path / "transform.log"

        args = argparse.Namespace(
            alignment_path=str(sample_transform_h5),
            image_path=str(sample_nifti_file),
            label_image=False,
            apply_preprocessing=False,
            result_name=None,
            log=str(log_path),
            verbose=True,
        )

        transform(args)

        assert log_path.exists()
        assert log_path.stat().st_size > 0


class TestTransformInputValidation:
    """Tests for input validation in transform command."""

    def test_transform_missing_h5_file(self, tmp_path: Path, sample_nifti_file: Path) -> None:
        """Test error when transform.h5 is missing."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        args = argparse.Namespace(
            alignment_path=str(empty_dir),
            image_path=str(sample_nifti_file),
            label_image=False,
            apply_preprocessing=False,
            result_name=None,
            log=None,
            verbose=True,
        )

        with pytest.raises(AssertionError):
            transform(args)

    def test_transform_missing_image(self, tmp_path: Path, sample_transform_h5):
        """Test error when input image is missing."""
        args = argparse.Namespace(
            alignment_path=str(sample_transform_h5),
            image_path=str(tmp_path / "nonexistent.nii"),
            label_image=False,
            apply_preprocessing=False,
            result_name=None,
            log=None,
            verbose=True,
        )

        # guarded_ants_image_read will raise an exception
        with pytest.raises(Exception) as exc_info:
            transform(args)


class TestTransformMetadataPreservation:
    """Tests for metadata preservation during transformation."""

    def test_transform_preserves_spacing(self, tmp_path: Path, sample_transform_h5: Path, sample_3d_image):
        """Test that output image has correct spacing from fixed image."""
        # Create image with custom spacing
        custom_spacing = (2.0, 2.0, 2.0)
        custom_img = ants.from_numpy(
            sample_3d_image.numpy(),
            origin=sample_3d_image.origin,
            spacing=custom_spacing,
            direction=sample_3d_image.direction,
            has_components=sample_3d_image.has_components,
        )

        image_path = tmp_path / "custom_spacing.nii"
        ants.image_write(custom_img, str(image_path))

        args = argparse.Namespace(
            alignment_path=str(sample_transform_h5),
            image_path=str(image_path),
            label_image=False,
            apply_preprocessing=False,
            result_name=None,
            log=None,
            verbose=True,
        )

        transform(args)

        result_name = f"{image_path.stem}_transformed.nii"
        result_path = sample_transform_h5 / result_name
        transformed = ants.image_read(str(result_path))

        # Should match the fixed image spacing from transform.h5
        with h5py.File(str(sample_transform_h5 / "transform.h5"), "r") as h5_handle:
            expected_spacing = tuple(h5_handle.attrs["fixed.spacing"])

        assert transformed.spacing == expected_spacing
        result_path.unlink()

    def test_transform_preserves_origin(self, tmp_path: Path, sample_transform_h5: Path, sample_nifti_file: Path):
        """Test that output image has correct origin from fixed image."""
        args = argparse.Namespace(
            alignment_path=str(sample_transform_h5),
            image_path=str(sample_nifti_file),
            label_image=False,
            apply_preprocessing=False,
            result_name=None,
            log=None,
            verbose=True,
        )

        transform(args)

        result_name = f"{sample_nifti_file.stem}_transformed.nii"
        result_path = sample_transform_h5 / result_name
        transformed = ants.image_read(str(result_path))

        # Should match the fixed image origin from transform.h5
        with h5py.File(str(sample_transform_h5 / "transform.h5"), "r") as h5_handle:
            expected_origin = tuple(h5_handle.attrs["fixed.origin"])

        assert transformed.origin == expected_origin
        result_path.unlink()
