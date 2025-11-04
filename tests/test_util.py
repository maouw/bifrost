"""Tests for bifrost utility functions."""

import numpy as np
import pytest

from bifrost.util import (
    dice_coefficient,
    sha256,
)

"""Tests for Dice coefficient calculation."""


def test_dice_coefficient_identical_images() -> None:
    """Test Dice coefficient for identical label images."""
    labels = np.zeros((50, 50, 50), dtype=np.int32)
    labels[10:20, 10:20, 10:20] = 1
    labels[30:40, 30:40, 30:40] = 2

    mean_coeff, label_coeffs = dice_coefficient(labels, labels)

    # Identical images should have Dice = 1.0
    assert mean_coeff == 1.0
    assert all(coeff == 1.0 for coeff in label_coeffs.values())


def test_dice_coefficient_no_overlap() -> None:
    """Test Dice coefficient for non-overlapping regions."""
    image1 = np.zeros((50, 50, 50), dtype=np.int32)
    image1[10:20, 10:20, 10:20] = 1

    image2 = np.zeros((50, 50, 50), dtype=np.int32)
    image2[30:40, 30:40, 30:40] = 1

    mean_coeff, label_coeffs = dice_coefficient(image1, image2)

    # No overlap should give Dice = 0.0
    assert label_coeffs[1] == 0.0


def test_dice_coefficient_partial_overlap() -> None:
    """Test Dice coefficient for partially overlapping regions."""
    image1 = np.zeros((50, 50, 50), dtype=np.int32)
    image1[10:30, 10:30, 10:30] = 1

    image2 = np.zeros((50, 50, 50), dtype=np.int32)
    image2[20:40, 20:40, 20:40] = 1

    mean_coeff, label_coeffs = dice_coefficient(image1, image2)

    # Partial overlap should give 0 < Dice < 1
    assert 0.0 < label_coeffs[1] < 1.0


def test_dice_coefficient_exclude_background() -> None:
    """Test Dice coefficient with background exclusion."""
    labels = np.zeros((50, 50, 50), dtype=np.int32)
    labels[10:20, 10:20, 10:20] = 1
    labels[30:40, 30:40, 30:40] = 2

    mean_coeff, label_coeffs = dice_coefficient(labels, labels, exclude_labels=[0])

    # Background (label 0) should not be in results
    assert 0 not in label_coeffs
    assert 1 in label_coeffs
    assert 2 in label_coeffs


def test_dice_coefficient_multiple_labels() -> None:
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


def test_dice_coefficient_mismatched_shapes() -> None:
    """Test error when image shapes don't match."""
    image1 = np.zeros((50, 50, 50), dtype=np.int32)
    image2 = np.zeros((40, 40, 40), dtype=np.int32)

    with pytest.raises(AssertionError, match="Shape mismatch"):
        dice_coefficient(image1, image2)


def test_dice_coefficient_mismatched_labels() -> None:
    """Test error when images have different label sets."""
    image1 = np.zeros((50, 50, 50), dtype=np.int32)
    image1[10:20, 10:20, 10:20] = 1

    image2 = np.zeros((50, 50, 50), dtype=np.int32)
    image2[10:20, 10:20, 10:20] = 2  # Different label

    with pytest.raises(AssertionError):
        dice_coefficient(image1, image2)


"""Tests for hashing utilities."""


def test_sha256_consistent() -> None:
    """Test SHA256 produces consistent results."""
    data = b"Test data for hashing"
    hash1 = sha256(data)
    hash2 = sha256(data)

    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA256 produces 64 hex characters


def test_sha256_different_inputs() -> None:
    """Test SHA256 produces different hashes for different inputs."""
    data1 = b"First data"
    data2 = b"Second data"

    hash1 = sha256(data1)
    hash2 = sha256(data2)

    assert hash1 != hash2


def test_sha256_empty_string() -> None:
    """Test SHA256 on empty string."""
    hash_value = sha256(b"")

    assert isinstance(hash_value, str)
    assert len(hash_value) == 64
    # SHA256 of empty string is a known value
    assert hash_value == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_unicode() -> None:
    """Test SHA256 on unicode data."""
    data = "Hello, 世界! 🌍".encode()
    hash_value = sha256(data)

    assert isinstance(hash_value, str)
    assert len(hash_value) == 64
