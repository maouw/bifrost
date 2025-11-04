"""Tests for bifrost build_template command."""

from pathlib import Path

import ants
import numpy as np
import pytest

from bifrost.cli.build_template import BuildTemplateArgs, build_template


def test_build_template_basic_workflow(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test basic template building workflow."""
    # Create multiple input images
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    image_paths = []
    for i in range(3):
        # Create slightly different versions of the image
        img_data = sample_3d_image.numpy() + np.random.randn(*sample_3d_image.shape) * 5
        img = ants.from_numpy(
            img_data,
            origin=sample_3d_image.origin,
            spacing=sample_3d_image.spacing,
            direction=sample_3d_image.direction,
            has_components=sample_3d_image.has_components,
        )
        img_path = input_dir / f"image_{i}.nii"
        ants.image_write(img, str(img_path))
        image_paths.append(img_path)

    output_dir = tmp_path / "output"

    args = BuildTemplateArgs(
        input=image_paths,
        output=output_dir,
        syn_steps=1,  # Minimal for faster testing
        affine_steps=1,
        gradient_step=0.2,
        verbose=True,
    )

    build_template(args)

    # Check outputs
    assert output_dir.exists()
    assert (output_dir / "template.nii").exists()

    # Verify the template is a valid image
    template = ants.image_read(str(output_dir / "template.nii"))
    assert template is not None
    assert template.shape == sample_3d_image.shape


def test_build_template_with_directory_input(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test template building with directory as input."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    for i in range(3):
        img_data = sample_3d_image.numpy() + np.random.randn(*sample_3d_image.shape) * 5
        img = ants.from_numpy(
            img_data,
            origin=sample_3d_image.origin,
            spacing=sample_3d_image.spacing,
            direction=sample_3d_image.direction,
            has_components=sample_3d_image.has_components,
        )
        ants.image_write(img, str(input_dir / f"image_{i}.nii"))

    output_dir = tmp_path / "output"

    args = BuildTemplateArgs(
        input=[input_dir],  # Pass directory instead of individual files
        output=output_dir,
        syn_steps=1,
        affine_steps=1,
        gradient_step=0.1,
        verbose=True,
    )

    build_template(args)

    assert (output_dir / "template.nii").exists()


def test_build_template_with_reference_image(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test template building with a specified reference image."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    image_paths = []
    for i in range(3):
        img_data = sample_3d_image.numpy() + np.random.randn(*sample_3d_image.shape) * 5
        img = ants.from_numpy(
            img_data,
            origin=sample_3d_image.origin,
            spacing=sample_3d_image.spacing,
            direction=sample_3d_image.direction,
            has_components=sample_3d_image.has_components,
        )
        img_path = input_dir / f"image_{i}.nii"
        ants.image_write(img, str(img_path))
        image_paths.append(img_path)

    # Use first image as reference
    reference_path = image_paths[0]
    output_dir = tmp_path / "output"

    args = BuildTemplateArgs(
        input=image_paths,
        output=output_dir,
        reference_image=Path(reference_path),
        affine_steps=1,
        syn_steps=1,
        gradient_step=0.1,
        verbose=True,
    )

    build_template(args)

    assert (output_dir / "template.nii").exists()


def test_build_template_with_clahe_preprocessing(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test template building with CLAHE preprocessing."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    image_paths = []
    for i in range(2):
        img_data = sample_3d_image.numpy() + np.random.randn(*sample_3d_image.shape) * 5
        img = ants.from_numpy(
            img_data,
            origin=sample_3d_image.origin,
            spacing=sample_3d_image.spacing,
            direction=sample_3d_image.direction,
            has_components=sample_3d_image.has_components,
        )
        img_path = input_dir / f"image_{i}.nii"
        ants.image_write(img, str(img_path))
        image_paths.append(img_path)

    output_dir = tmp_path / "output"

    args = BuildTemplateArgs(
        input=image_paths,
        output=output_dir,
        affine_steps=1,
        syn_steps=2,
        gradient_step=0.2,
        preprocessing=("CLAHE",),
        keep_intermediates=True,
        verbose=True,
    )

    build_template(args)

    assert (output_dir / "template.nii").exists()

    # Check that preprocessed images were created
    assert (output_dir / "preprocessed").exists()


def test_build_template_with_mirror(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test template building with mirroring enabled."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    image_paths = []
    for i in range(2):
        img_data = sample_3d_image.numpy() + np.random.randn(*sample_3d_image.shape) * 5
        img = ants.from_numpy(
            img_data,
            origin=sample_3d_image.origin,
            spacing=sample_3d_image.spacing,
            direction=sample_3d_image.direction,
            has_components=sample_3d_image.has_components,
        )
        img_path = input_dir / f"image_{i}.nii"
        ants.image_write(img, str(img_path))
        image_paths.append(img_path)

    output_dir = tmp_path / "output"

    args = BuildTemplateArgs(
        input=image_paths,
        output=output_dir,
        affine_steps=1,
        syn_steps=1,
        gradient_step=0.1,
        mirror=True,
        verbose=True,
    )

    build_template(args)

    assert (output_dir / "template.nii").exists()


def test_build_template_keep_intermediates(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test that intermediate results are kept when requested."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    image_paths = []
    for i in range(2):
        img_data = sample_3d_image.numpy() + np.random.randn(*sample_3d_image.shape) * 5
        img = ants.from_numpy(
            img_data,
            origin=sample_3d_image.origin,
            spacing=sample_3d_image.spacing,
            direction=sample_3d_image.direction,
            has_components=sample_3d_image.has_components,
        )
        img_path = input_dir / f"image_{i}.nii"
        ants.image_write(img, str(img_path))
        image_paths.append(img_path)

    output_dir = tmp_path / "output"

    args = BuildTemplateArgs(
        input=image_paths,
        output=output_dir,
        affine_steps=1,
        syn_steps=1,
        gradient_step=0.1,
        keep_intermediates=True,
        verbose=True,
    )

    build_template(args)

    # Check that intermediate directories still exist
    assert (output_dir / "preprocessed").exists()
    assert (output_dir / "templates").exists()


def test_build_template_multiple_affine_steps(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test template building with multiple affine steps."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    image_paths = []
    for i in range(2):
        img_data = sample_3d_image.numpy() + np.random.randn(*sample_3d_image.shape) * 5
        img = ants.from_numpy(
            img_data,
            origin=sample_3d_image.origin,
            spacing=sample_3d_image.spacing,
            direction=sample_3d_image.direction,
            has_components=sample_3d_image.has_components,
        )
        img_path = input_dir / f"image_{i}.nii"
        ants.image_write(img, str(img_path))
        image_paths.append(img_path)

    output_dir = tmp_path / "output"

    args = BuildTemplateArgs(
        input=image_paths,
        output=output_dir,
        affine_steps=2,
        syn_steps=1,
        gradient_step=0.1,
        keep_intermediates=True,
        verbose=True,
    )

    build_template(args)

    assert (output_dir / "template.nii").exists()

    # Check that intermediate templates were created
    assert (output_dir / "templates" / "affine_0.nii").exists()
    assert (output_dir / "templates" / "affine_1.nii").exists()


def test_build_template_force_overwrite(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test force overwrite of existing output directory."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    image_paths = []
    for i in range(2):
        img_data = sample_3d_image.numpy() + np.random.randn(*sample_3d_image.shape) * 5
        img = ants.from_numpy(
            img_data,
            origin=sample_3d_image.origin,
            spacing=sample_3d_image.spacing,
            direction=sample_3d_image.direction,
            has_components=sample_3d_image.has_components,
        )
        img_path = input_dir / f"image_{i}.nii"
        ants.image_write(img, str(img_path))
        image_paths.append(img_path)

    output_dir = tmp_path / "output"

    # First run
    args = BuildTemplateArgs(
        input=image_paths, output=output_dir, affine_steps=1, syn_steps=1, gradient_step=0.1, verbose=True
    )
    build_template(args)
    first_template_mtime = (output_dir / "template.nii").stat().st_mtime

    # Second run witqhout force (should skip)
    build_template(args)
    unchanged_mtime = (output_dir / "template.nii").stat().st_mtime
    assert first_template_mtime == unchanged_mtime

    # Third run with force
    args.force = True
    build_template(args)
    new_mtime = (output_dir / "template.nii").stat().st_mtime
    assert new_mtime >= first_template_mtime


def test_build_template_custom_log(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test template building with custom log file."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    image_paths = []
    for i in range(2):
        img_data = sample_3d_image.numpy() + np.random.randn(*sample_3d_image.shape) * 5
        img = ants.from_numpy(
            img_data,
            origin=sample_3d_image.origin,
            spacing=sample_3d_image.spacing,
            direction=sample_3d_image.direction,
            has_components=sample_3d_image.has_components,
        )
        img_path = input_dir / f"image_{i}.nii"
        ants.image_write(img, str(img_path))
        image_paths.append(img_path)

    output_dir = tmp_path / "output"
    log_path = tmp_path / "custom.log"

    args = BuildTemplateArgs(
        input=image_paths,
        output=output_dir,
        affine_steps=1,
        syn_steps=2,
        gradient_step=0.2,
        log=log_path,
        verbose=True,
    )

    build_template(args)

    assert log_path.exists()
    assert log_path.stat().st_size > 0


def test_build_template_missing_input(tmp_path: Path) -> None:
    """Test error when input files don't exist."""
    output_dir = tmp_path / "output"

    args = BuildTemplateArgs(
        input=[tmp_path / "nonexistent.nii"], output=output_dir, affine_steps=1, syn_steps=1, gradient_step=0.1
    )

    with pytest.raises(AssertionError):
        build_template(args)


def test_build_template_missing_reference(tmp_path: Path, sample_3d_image: ants.ANTsImage) -> None:
    """Test error when reference image doesn't exist."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    img_path = input_dir / "image.nii"
    ants.image_write(sample_3d_image, str(img_path))

    output_dir = tmp_path / "output"

    args = BuildTemplateArgs(
        input=[img_path],
        output=output_dir,
        reference_image=tmp_path / "nonexistent_reference.nii",
        affine_steps=1,
        syn_steps=1,
        gradient_step=0.1,
    )

    with pytest.raises(AssertionError):
        build_template(args)


def test_build_template_output_exists_no_force(
    tmp_path: Path, sample_3d_image: ants.ANTsImage, caplog: pytest.LogCaptureFixture
) -> None:
    """Test warning when output directory exists without force flag."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    image_paths = []
    for i in range(2):
        img_path = input_dir / f"image_{i}.nii"
        ants.image_write(sample_3d_image, str(img_path))
        image_paths.append(img_path)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    args = BuildTemplateArgs(
        input=image_paths, output=output_dir, affine_steps=1, syn_steps=1, gradient_step=0.1, verbose=True
    )

    build_template(args)

    # Should exit early without creating template
    assert not (output_dir / "template.nii").exists()
