"""Misc. utility methods."""

import argparse
import hashlib
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import ants
import numpy as np
from ants import ANTsImage
from numpy import ndarray

from bifrost.types import Pathable

SYNTHMORPH_SHAPE = (160, 160, 192)


def update_image_array(image: ANTsImage, updated: ANTsImage | ndarray) -> ANTsImage:
    """Update ANTsImage image array but preserve metadata.

    Args:
        image: the image to update
        updated: array to replace image data with

    Returns:
        updated_image: the updated ANTsImage
    """
    assert image.numpy().shape == updated.shape
    # assert image.shape == updated.shape, f"Shape mismatch: {image.shape} != {updated.shape}. Ensure the updated array has the same shape as the original image."
    if not isinstance(updated, np.ndarray):
        updated = updated.numpy()
    return ants.new_image_like(image, updated)


def threshold_image(image: ANTsImage, threshold: float) -> ANTsImage:
    """Set intensity values below threshold to 0.

    Args:
        image: ANTsImage
        threshold: float

    Returns:
        thresholded_img: ANTsImage
    """
    image_arr = image.numpy()

    image_arr[image_arr <= threshold] = 0.0

    return ants.from_numpy(
        image_arr,
        origin=image.origin,
        spacing=image.spacing,
        direction=image.direction,
        has_components=image.has_components,
    )


def transpose_image(image: ANTsImage, transposition: ndarray) -> ANTsImage:
    """Transpose the axes of an image, preserve metadata.

    Args:
    image: image to transpose
    transposition: permutation of axes, same format as np.transpose

    Returns:
        transposed_image: the transposed image
    """
    # assert sorted(transposition) == list(range(len(transposition)))
    np.testing.assert_array_equal(np.sort(transposition), np.arange(len(transposition)))

    def _permute(arr):
        # return arr[transposition]
        return [arr[idx] for idx in transposition]
        # return np.array(arr)[transposition]

    image_arr = np.transpose(image.numpy(), transposition)

    return ants.from_numpy(
        image_arr,
        origin=_permute(image.origin),
        spacing=_permute(image.spacing),
        direction=np.stack(_permute(image.direction)),
        has_components=image.has_components,
    )


def dice_coefficient(image_1: ndarray, image_2: ndarray, exclude_labels: set[int] | Sequence[int] = (0,)):
    """Computes the mean Sørensen-Dice coefficient across labels for two images.

    Also returns per label dice coefficients.

    Args:
        image_1: an image array
        image_2: an image array
        exclude_labels: (optional) list of labels to exclude, say the background - list

    Returns:
        mean_coeff: float
        label_coeffs: map of label to dice coeff - dict
    """
    assert image_1.shape == image_2.shape, (
        f"Shape mismatch: {image_1.shape} != {image_2.shape}. Ensure both images have the same shape."
    )

    labels = np.unique(image_1)
    np.testing.assert_allclose(labels, np.unique(image_2))  # assert all(labels == np.unique(image_2))

    label_coeffs = {}

    exclude_labels = set(exclude_labels)
    for label in labels:
        assert float(label).is_integer()

        if label not in exclude_labels:
            mask_1 = image_1 == label
            mask_2 = image_2 == label
            label_coeffs[int(label)] = 2 * np.sum(mask_1 * mask_2) / (np.sum(mask_1) + np.sum(mask_2))

    mean_coeff = np.mean(list(label_coeffs.values()))

    return mean_coeff, label_coeffs


def sha256(byte_string):
    """Returns the hex sha256 digest of a byte encoded string."""
    digester = hashlib.sha256()
    digester.update(byte_string)
    return digester.hexdigest()


class SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Adapted from https://stackoverflow.com/questions/13423540/argparse-subparser-hide-metavar-in-command-listing."""

    def _format_action(self, action):
        parts = super(argparse.RawDescriptionHelpFormatter, self)._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts


def default_arg_from_env_var(env_var, value_name="default"):
    """Returns a default argument from an environment variable for use in argparse.

    Args:
        env_var: the environment variable to read
        value_name: name of the value to return in the dictionary
    Returns:
        A dictionary with the value_name as key and the environment variable value as value, or an empty dictionary if the variable is not set.
    """
    v = os.environ.get(env_var)
    return {value_name: v} if v else {}


def find_images(
    directory: Pathable,
    extensions: str | tuple[str, ...] = (".nii", ".nii.gz"),
    max_depth: int = -1,
) -> list[Pathable]:
    """List files in a directory with given extensions, up to a maximum depth.

    Args:
        directory: path to the directory
        extensions: file extension or list of file extensions to include. Empty string for all files.
        max_depth: maximum depth to search for files. 0 for only top-level, None for unlimited depth.

    Returns:
        List of Paths to files with the given extensions.
    """
    extensions = (extensions,) if isinstance(extensions, str) else tuple(extensions)
    files = []

    def _gather_files_at_depth(current_path: Path, current_depth: int):
        # Base case: if current depth exceeds max depth, return
        if max_depth < 0:  # unlimited depth
            pass
        elif current_depth > max_depth:  # exceeded max depth
            return
        for item in current_path.iterdir():
            if item.is_file() and item.name.endswith(extensions):
                files.append(item)
            elif item.is_dir():
                _gather_files_at_depth(item, current_depth + 1)

    _gather_files_at_depth(Path(directory), 0)
    return files


def setup_cli_logger(
    name: str,
    verbose: bool = False,
    log_path: Pathable = None,
    default_log_name: Pathable = None,
) -> logging.Logger:
    """Set up logger for CLI commands with consistent configuration.

    Creates a logger with both console (stdout/stderr) and file handlers.
    Console output respects the verbose flag and BIFROST_LOG_LEVEL environment variable.

    Args:
        name: Logger name (typically __name__)
        verbose: If True, show INFO level messages on stdout. If False, only show errors.
        log_path: Explicit path to log file. Takes precedence over default_log_name.
        default_log_name: Default log file name/path if log_path not specified.

    Returns:
        Configured logger instance

    Raises:
        SystemExit: If BIFROST_LOG_LEVEL environment variable contains invalid value
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Default level, overridden by handlers

    # stdout handler (info messages only, no warnings/errors)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    logger.addHandler(stdout_handler)

    # stderr handler (warnings and errors only)
    error_handler = logging.StreamHandler(stream=sys.stderr)
    error_handler.setLevel(logging.WARNING)
    logger.addHandler(error_handler)

    # Configure verbosity
    if verbose:
        stdout_handler.setLevel(logging.INFO)
    else:
        stdout_handler.setLevel(logging.CRITICAL + 1)  # Suppress all stdout

    # Environment variable override for log level
    if env_level := os.environ.get("BIFROST_LOG_LEVEL"):
        logger.info("Overriding log level with BIFROST_LOG_LEVEL: %s", env_level)
        try:
            log_level = getattr(logging, env_level.upper())
            logger.setLevel(log_level)
            stdout_handler.setLevel(log_level)
            error_handler.setLevel(log_level)
        except AttributeError:
            logger.error("Invalid log level specified in BIFROST_LOG_LEVEL: %s", env_level)
            sys.exit(1)

    # File handler (if log path specified)
    file_log_path = log_path or default_log_name
    if file_log_path:
        setup_cli_file_logger(logger, file_log_path)
    return logger


def setup_cli_file_logger(logger: logging.Logger, log_path: Pathable) -> None:
    """Add file handler to existing logger instance."""
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Writing full logs to %s", log_path)


def images_are_conformable(
    image1: ants.ANTsImage, image2: ants.ANTsImage, shape: bool = True, spacing: bool = True, direction: bool = True
) -> bool:
    """Check if two ANTsImages are conformable (i.e., have the same shape and spacing).

    Args:
        image1: the first image
        image2: the second image
        shape: whether to check shape conformity
        spacing: whether to check spacing conformity
        direction: whether to check direction conformity

    Returns:
        True if the images are conformable, False otherwise.
    """
    if shape and image1.shape != image2.shape:
        return False
    if spacing and image1.spacing != image2.spacing:
        return False
    if direction and not np.array_equal(image1.direction, image2.direction):  # noqa: SIM103
        return False
    return True
