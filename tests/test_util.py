"""Tests for bifrost utility functions."""

import argparse
import logging
import os
import sys
from pathlib import Path

import ants
import numpy as np
import pytest

from bifrost.util import (
    SubcommandHelpFormatter,
    default_arg_from_env_var,
    dice_coefficient,
    find_images,
    setup_cli_file_logger,
    setup_cli_logger,
    sha256,
    threshold_image,
    transpose_image,
    update_image_array,
)

if "BIFROST_LOG_LEVEL" in os.environ:
    del os.environ["BIFROST_LOG_LEVEL"]


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


class TestFindImages:
    """Tests for find_images file discovery."""

    def test_find_images_basic(self, tmp_path) -> None:
        """Test finding .nii files in directory."""
        # Create test files
        (tmp_path / "image1.nii").touch()
        (tmp_path / "image2.nii").touch()
        (tmp_path / "other.txt").touch()

        files = find_images(tmp_path, extensions=".nii")

        assert len(files) == 2
        assert all(str(f).endswith(".nii") for f in files)

    def test_find_images_multiple_extensions(self, tmp_path) -> None:
        """Test finding files with multiple extensions."""
        (tmp_path / "image1.nii").touch()
        (tmp_path / "image2.nii.gz").touch()
        (tmp_path / "image3.tif").touch()

        files = find_images(tmp_path, extensions=(".nii", ".nii.gz"))

        assert len(files) == 2

    def test_find_images_nested_directories(self, tmp_path) -> None:
        """Test finding files in nested directories."""
        (tmp_path / "image1.nii").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "image2.nii").touch()
        (subdir / "subsubdir").mkdir()
        (subdir / "subsubdir" / "image3.nii").touch()

        files = find_images(tmp_path, extensions=".nii")

        assert len(files) == 3

    def test_find_images_max_depth_zero(self, tmp_path) -> None:
        """Test max_depth=0 only searches top level."""
        (tmp_path / "image1.nii").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "image2.nii").touch()

        files = find_images(tmp_path, extensions=".nii", max_depth=0)

        assert len(files) == 1
        assert files[0].name == "image1.nii"

    def test_find_images_max_depth_one(self, tmp_path) -> None:
        """Test max_depth=1 searches one level deep."""
        (tmp_path / "image1.nii").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "image2.nii").touch()
        (subdir / "subsubdir").mkdir()
        (subdir / "subsubdir" / "image3.nii").touch()

        files = find_images(tmp_path, extensions=".nii", max_depth=1)

        assert len(files) == 2
        assert not any("subsubdir" in str(f) for f in files)

    def test_find_images_empty_directory(self, tmp_path) -> None:
        """Test finding images in empty directory."""

        files = find_images(tmp_path, extensions=".nii")

        assert len(files) == 0

    def test_find_images_no_matching_files(self, tmp_path) -> None:
        """Test when no files match extension."""
        (tmp_path / "image1.txt").touch()
        (tmp_path / "image2.jpg").touch()

        files = find_images(tmp_path, extensions=".nii")

        assert len(files) == 0


class TestDefaultArgFromEnvVar:
    """Tests for default_arg_from_env_var."""

    def test_default_arg_from_env_var_present(self, monkeypatch) -> None:
        """Test reading environment variable when present."""

        monkeypatch.setenv("TEST_VAR", "/path/to/file")

        result = default_arg_from_env_var("TEST_VAR")

        assert result == {"default": "/path/to/file"}

    def test_default_arg_from_env_var_absent(self) -> None:
        """Test reading environment variable when absent."""

        result = default_arg_from_env_var("NONEXISTENT_VAR")

        assert result == {}

    def test_default_arg_from_env_var_custom_key(self, monkeypatch) -> None:
        """Test custom value_name parameter."""

        monkeypatch.setenv("TEST_VAR", "value")

        result = default_arg_from_env_var("TEST_VAR", value_name="custom")

        assert result == {"custom": "value"}


class TestUpdateImageArrayWithAntsImage:
    """Additional tests for update_image_array with ANTsImage input."""

    def test_update_image_array_with_ants_image(self, sample_3d_image: ants.ANTsImage) -> None:
        """Test updating image with another ANTsImage."""
        new_image = sample_3d_image.clone()
        new_data = new_image.numpy() * 3.0
        new_image = ants.from_numpy(new_data, origin=new_image.origin, spacing=new_image.spacing)

        updated_image = update_image_array(sample_3d_image, new_image)

        np.testing.assert_array_equal(updated_image.numpy(), new_data)
        assert updated_image.origin == sample_3d_image.origin
        assert updated_image.spacing == sample_3d_image.spacing


class TestSubcommandHelpFormatter:
    """Tests for SubcommandHelpFormatter."""

    def test_subcommand_help_formatter_format_action(self) -> None:
        """Test that SubcommandHelpFormatter correctly formats subparser actions."""

        parser = argparse.ArgumentParser(formatter_class=SubcommandHelpFormatter)
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Add a subcommand
        subparsers.add_parser("test_command", help="Test command help")

        # Get the formatted help
        help_output = parser.format_help()

        # The formatter should remove the first line of subparser actions (the metavar line)
        # Check that the help is formatted (we can't easily test exact format without implementation details)
        assert "test_command" in help_output
        assert isinstance(help_output, str)

    def test_subcommand_help_formatter_with_raw_description(self) -> None:
        """Test that SubcommandHelpFormatter preserves raw description formatting."""

        description = "Line 1\n    Line 2 with indent\nLine 3"
        parser = argparse.ArgumentParser(formatter_class=SubcommandHelpFormatter, description=description)

        help_output = parser.format_help()

        # RawDescriptionHelpFormatter should preserve the description formatting
        assert "Line 1" in help_output
        assert "Line 2 with indent" in help_output


class TestSetupCliLogger:
    """Tests for setup_cli_logger."""

    def test_setup_cli_logger_default(self) -> None:
        """Test logger setup with default parameters."""
        logger = setup_cli_logger("test_logger")

        assert logger.name == "test_logger"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 2  # stdout and stderr handlers

    def test_setup_cli_logger_verbose(self) -> None:
        """Test logger setup with verbose=True."""
        logger = setup_cli_logger("test_logger_verbose", verbose=True)

        # Find stdout handler
        stdout_handler = next(h for h in logger.handlers if h.stream == sys.stdout)
        assert stdout_handler.level == logging.INFO

    def test_setup_cli_logger_not_verbose(self) -> None:
        """Test logger setup with verbose=False."""
        logger = setup_cli_logger("test_logger_quiet", verbose=False)

        # Find stdout handler
        stdout_handler = next(h for h in logger.handlers if h.stream == sys.stdout)
        assert stdout_handler.level != logging.INFO

    def test_setup_cli_logger_with_log_path(self, tmp_path) -> None:
        """Test logger setup with explicit log path."""
        log_file = tmp_path / "test.log"

        logger = setup_cli_logger("test_logger_file", log_path=log_file)

        # Should have 3 handlers: stdout, stderr, file
        assert len(logger.handlers) == 3
        file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        assert file_handler.level == logging.DEBUG
        assert log_file.exists()

    def test_setup_cli_logger_with_default_log_name(self, tmp_path) -> None:
        """Test logger setup with default_log_name."""
        log_file = tmp_path / "default.log"

        logger = setup_cli_logger("test_logger_default", default_log_name=log_file)

        assert len(logger.handlers) == 3
        assert log_file.exists()

    def test_setup_cli_logger_log_path_precedence(self, tmp_path) -> None:
        """Test that log_path takes precedence over default_log_name."""
        log_path_file = tmp_path / "explicit.log"
        default_file = tmp_path / "default.log"

        logger = setup_cli_logger("test_logger_precedence", log_path=log_path_file, default_log_name=default_file)

        assert log_path_file.exists()
        assert not default_file.exists()

    def test_setup_cli_logger_env_var_override(self, monkeypatch) -> None:
        """Test BIFROST_LOG_LEVEL environment variable override."""
        monkeypatch.setenv("BIFROST_LOG_LEVEL", "WARNING")

        logger = setup_cli_logger("test_logger_env", verbose=True)

        assert logger.level == logging.WARNING
        # All handlers should be set to WARNING
        for handler in logger.handlers:
            if handler.stream in (sys.stdout, sys.stderr):
                assert handler.level == logging.WARNING

    def test_setup_cli_logger_invalid_env_var(self, monkeypatch, capsys) -> None:
        """Test invalid BIFROST_LOG_LEVEL value exits with error."""
        monkeypatch.setenv("BIFROST_LOG_LEVEL", "INVALID_LEVEL")

        with pytest.raises(SystemExit) as exc_info:
            setup_cli_logger("test_logger_invalid")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Invalid log level" in captured.err

    def test_setup_cli_logger_stderr_handler(self) -> None:
        """Test stderr handler only logs warnings and errors."""
        logger = setup_cli_logger("test_logger_stderr")

        stderr_handler = next(h for h in logger.handlers if h.stream == sys.stderr)
        assert stderr_handler.level == logging.WARNING

    def test_setup_cli_logger_stdout_filter(self) -> None:
        """Test stdout handler filters out warnings and errors."""
        logger = setup_cli_logger("test_logger_stdout_filter", verbose=True)

        stdout_handler = next(h for h in logger.handlers if h.stream == sys.stdout)

        # Create test records
        info_record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        warning_record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )

        # stdout should accept INFO but filter WARNING
        assert stdout_handler.filter(info_record)
        assert not stdout_handler.filter(warning_record)


class TestSetupCliFileLogger:
    """Tests for setup_cli_file_logger."""

    def test_setup_cli_file_logger(self, tmp_path) -> None:
        """Test adding file handler to existing logger."""
        logger = logging.getLogger("test_file_logger")
        logger.setLevel(logging.DEBUG)
        log_file = tmp_path / "test.log"

        setup_cli_file_logger(logger, log_file)

        # Should have file handler added
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.DEBUG
        assert log_file.exists()

    def test_setup_cli_file_logger_formatter(self, tmp_path) -> None:
        """Test file handler has correct formatter."""
        logger = logging.getLogger("test_file_formatter")
        log_file = tmp_path / "test.log"

        setup_cli_file_logger(logger, log_file)

        file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        formatter = file_handler.formatter

        assert formatter is not None
        # Test formatter format string contains expected components
        assert "asctime" in formatter._fmt
        assert "name" in formatter._fmt
        assert "levelname" in formatter._fmt
        assert "message" in formatter._fmt

    def test_setup_cli_file_logger_writes_message(self, tmp_path) -> None:
        """Test file logger actually writes to file."""
        logger = logging.getLogger("test_file_write")
        logger.setLevel(logging.DEBUG)
        log_file = tmp_path / "test.log"

        setup_cli_file_logger(logger, log_file)
        logger.info("Test message")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Test message" in log_content
        assert "INFO" in log_content

    def test_setup_cli_file_logger_pathable_types(self, tmp_path) -> None:
        """Test file logger accepts different Pathable types."""
        logger = logging.getLogger("test_pathable")

        # Test with Path object
        log_file_path = tmp_path / "path.log"
        setup_cli_file_logger(logger, log_file_path)
        assert log_file_path.exists()

        # Test with string
        logger2 = logging.getLogger("test_pathable2")
        log_file_str = str(tmp_path / "string.log")
        setup_cli_file_logger(logger2, log_file_str)
        assert Path(log_file_str).exists()
