"""Tests for bifrost utility functions."""

import ants
import numpy as np
import pytest

from bifrost.util import (
    dice_coefficient,
    sha256,
    threshold_image,
    transpose_image,
    update_image_array,
)


class TestImageUtilities:
    """Tests for image manipulation utilities."""

    def test_update_image_array(self, sample_3d_image: ants.ANTsImage) -> None:
        """Test updating image array while preserving metadata."""
        original_data = sample_3d_image.numpy()
        new_data = original_data * 2.0

        updated_image = update_image_array(sample_3d_image, new_data)

        # Data should be updated
        np.testing.assert_array_equal(updated_image.numpy(), new_data)

        # Metadata should be preserved
        assert updated_image.origin == sample_3d_image.origin
        assert updated_image.spacing == sample_3d_image.spacing
        np.testing.assert_array_equal(updated_image.direction, sample_3d_image.direction)
        assert updated_image.has_components == sample_3d_image.has_components

    def test_update_image_array_wrong_shape(self, sample_3d_image: ants.ANTsImage) -> None:
        """Test error when array has wrong shape."""
        wrong_shape_data = np.random.rand(10, 10, 10)

        with pytest.raises(AssertionError):
            update_image_array(sample_3d_image, wrong_shape_data)

    def test_threshold_image(self, sample_3d_image: ants.ANTsImage) -> None:
        """Test thresholding image intensity values."""
        threshold = 50.0
        thresholded = threshold_image(sample_3d_image, threshold)

        # Values below threshold should be zero
        original_data = sample_3d_image.numpy()
        thresholded_data = thresholded.numpy()

        below_threshold_mask = original_data <= threshold
        assert np.all(thresholded_data[below_threshold_mask] == 0.0)

        # Values above threshold should be preserved
        above_threshold_mask = original_data > threshold
        np.testing.assert_array_equal(thresholded_data[above_threshold_mask], original_data[above_threshold_mask])

        # Metadata should be preserved
        assert thresholded.origin == sample_3d_image.origin
        assert thresholded.spacing == sample_3d_image.spacing

    def test_transpose_image(self, sample_3d_image: ants.ANTsImage) -> None:
        """Test transposing image axes."""
        # Transpose: (0, 1, 2) -> (2, 0, 1)
        transposition = np.array([2, 0, 1])

        transposed = transpose_image(sample_3d_image, transposition)

        # Check shape is correctly transposed
        original_shape = sample_3d_image.shape
        expected_shape = tuple(original_shape[i] for i in transposition)
        assert transposed.shape == expected_shape

        # Check data is correctly transposed
        np.testing.assert_array_equal(transposed.numpy(), np.transpose(sample_3d_image.numpy(), transposition))

        # Check metadata is correctly permuted
        original_spacing = sample_3d_image.spacing
        expected_spacing = tuple(original_spacing[i] for i in transposition)
        assert transposed.spacing == expected_spacing

    def test_transpose_image_invalid_permutation(self, sample_3d_image: ants.ANTsImage) -> None:
        """Test error with invalid permutation."""
        invalid_transposition = np.array([0, 1, 3])  # Invalid: missing 2, has 3

        with pytest.raises(AssertionError):
            transpose_image(sample_3d_image, invalid_transposition)

    def test_transpose_image_identity(self, sample_3d_image: ants.ANTsImage) -> None:
        """Test identity transposition leaves image unchanged."""
        identity_transposition = np.array([0, 1, 2])

        transposed = transpose_image(sample_3d_image, identity_transposition)

        np.testing.assert_array_equal(transposed.numpy(), sample_3d_image.numpy())
        assert transposed.shape == sample_3d_image.shape
        assert transposed.spacing == sample_3d_image.spacing


class TestDiceCoefficient:
    """Tests for Dice coefficient calculation."""

    def test_dice_coefficient_identical_images(self) -> None:
        """Test Dice coefficient for identical label images."""
        labels = np.zeros((50, 50, 50), dtype=np.int32)
        labels[10:20, 10:20, 10:20] = 1
        labels[30:40, 30:40, 30:40] = 2

        mean_coeff, label_coeffs = dice_coefficient(labels, labels)

        # Identical images should have Dice = 1.0
        assert mean_coeff == 1.0
        assert all(coeff == 1.0 for coeff in label_coeffs.values())

    def test_dice_coefficient_no_overlap(self) -> None:
        """Test Dice coefficient for non-overlapping regions."""
        image1 = np.zeros((50, 50, 50), dtype=np.int32)
        image1[10:20, 10:20, 10:20] = 1

        image2 = np.zeros((50, 50, 50), dtype=np.int32)
        image2[30:40, 30:40, 30:40] = 1

        mean_coeff, label_coeffs = dice_coefficient(image1, image2)

        # No overlap should give Dice = 0.0
        assert label_coeffs[1] == 0.0

    def test_dice_coefficient_partial_overlap(self) -> None:
        """Test Dice coefficient for partially overlapping regions."""
        image1 = np.zeros((50, 50, 50), dtype=np.int32)
        image1[10:30, 10:30, 10:30] = 1

        image2 = np.zeros((50, 50, 50), dtype=np.int32)
        image2[20:40, 20:40, 20:40] = 1

        mean_coeff, label_coeffs = dice_coefficient(image1, image2)

        # Partial overlap should give 0 < Dice < 1
        assert 0.0 < label_coeffs[1] < 1.0

    def test_dice_coefficient_exclude_background(self) -> None:
        """Test Dice coefficient with background exclusion."""
        labels = np.zeros((50, 50, 50), dtype=np.int32)
        labels[10:20, 10:20, 10:20] = 1
        labels[30:40, 30:40, 30:40] = 2

        mean_coeff, label_coeffs = dice_coefficient(labels, labels, exclude_labels=[0])

        # Background (label 0) should not be in results
        assert 0 not in label_coeffs
        assert 1 in label_coeffs
        assert 2 in label_coeffs

    def test_dice_coefficient_multiple_labels(self) -> None:
        """Test Dice coefficient with multiple labels."""
        image1 = np.zeros((50, 50, 50), dtype=np.int32)
        image1[10:20, 10:20, 10:20] = 1
        image1[25:35, 25:35, 25:35] = 2

        image2 = image1.copy()
        # Slightly shift label 2
        image2[25:35, 25:35, 25:35] = 0
        image2[26:36, 26:36, 26:36] = 2

        mean_coeff, label_coeffs = dice_coefficient(image1, image2, exclude_labels=[0])

        # Label 1 should be perfect
        assert label_coeffs[1] == 1.0
        # Label 2 should have partial overlap
        assert 0.0 < label_coeffs[2] < 1.0
        # Mean should be between them
        assert label_coeffs[2] < mean_coeff < 1.0

    def test_dice_coefficient_mismatched_shapes(self) -> None:
        """Test error when image shapes don't match."""
        image1 = np.zeros((50, 50, 50), dtype=np.int32)
        image2 = np.zeros((40, 40, 40), dtype=np.int32)

        with pytest.raises(AssertionError, match="Shape mismatch"):
            dice_coefficient(image1, image2)

    def test_dice_coefficient_mismatched_labels(self) -> None:
        """Test error when images have different label sets."""
        image1 = np.zeros((50, 50, 50), dtype=np.int32)
        image1[10:20, 10:20, 10:20] = 1

        image2 = np.zeros((50, 50, 50), dtype=np.int32)
        image2[10:20, 10:20, 10:20] = 2  # Different label

        with pytest.raises(AssertionError):
            dice_coefficient(image1, image2)


class TestHashingUtilities:
    """Tests for hashing utilities."""

    def test_sha256_consistent(self) -> None:
        """Test SHA256 produces consistent results."""
        data = b"Test data for hashing"
        hash1 = sha256(data)
        hash2 = sha256(data)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 produces 64 hex characters

    def test_sha256_different_inputs(self) -> None:
        """Test SHA256 produces different hashes for different inputs."""
        data1 = b"First data"
        data2 = b"Second data"

        hash1 = sha256(data1)
        hash2 = sha256(data2)

        assert hash1 != hash2

    def test_sha256_empty_string(self) -> None:
        """Test SHA256 on empty string."""
        hash_value = sha256(b"")

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64
        # SHA256 of empty string is a known value
        assert hash_value == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_sha256_unicode(self) -> None:
        """Test SHA256 on unicode data."""
        data = "Hello, 世界! 🌍".encode()
        hash_value = sha256(data)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64
