"""Misc. utility methods."""

import argparse
import hashlib
import os
from collections.abc import Sequence

import ants
import numpy as np
import numpy.typing as npt

SYNTHMORPH_SHAPE = (160, 160, 192)


def update_image_array(image: ants.ANTsImage, updated: npt.NDArray) -> ants.ANTsImage:
    """Update ANTsImage image array but preserve metadata.

    Args:
        image: the image to update
        updated: array to replace image data with

    Returns:
        updated_image: the updated ANTsImage
    """
    assert image.shape == updated.shape, f"Shape mismatch: {image.shape} != {updated.shape}. Ensure the updated array has the same shape as the original image."

    updated_image = ants.from_numpy(
        updated,
        origin=image.origin,
        spacing=image.spacing,
        direction=image.direction,
        has_components=image.has_components,
    )

    return updated_image


def threshold_image(image: ants.ANTsImage, threshold: float) -> ants.ANTsImage:
    """Set intensity values below threshold to 0.

    Args:
        image: ants.ANTsImage
        threshold: float

    Returns:
        thresholded_img: ants.ANTsImage
    """
    image_arr = image.numpy()

    image_arr[image_arr <= threshold] = 0.0

    thresholded_image = ants.from_numpy(
        image_arr,
        origin=image.origin,
        spacing=image.spacing,
        direction=image.direction,
        has_components=image.has_components,
    )

    return thresholded_image


def transpose_image(image: ants.ANTsImage, transposition: npt.NDArray) -> ants.ANTsImage:
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

    transposed_image = ants.from_numpy(
        image_arr,
        origin=_permute(image.origin),
        spacing=_permute(image.spacing),
        direction=np.stack(_permute(image.direction)),
        has_components=image.has_components,
    )

    return transposed_image


def dice_coefficient(image_1: npt.NDArray, image_2: npt.NDArray, exclude_labels: set[int] | Sequence[int] = (0,)):
    """Computes the mean Sørensen–Dice coefficient across labels for two images.

    Also returns per label dice coefficients.

    Args:
        image_1: an image array
        image_2: an image array
        exclude_labels: (optional) list of labels to exclude, say the background - list

    Returns:
        mean_coeff: float
        label_coeffs: map of label to dice coeff - dict
    """
    assert image_1.shape == image_2.shape, f"Shape mismatch: {image_1.shape} != {image_2.shape}. Ensure both images have the same shape."

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
