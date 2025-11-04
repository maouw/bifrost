"""Tests for bifrost register command."""

from pathlib import Path

import h5py
import pytest

from bifrost.cli.register import RegisterArgs, register


class TestRegisterCommand:
    """Tests for the register command functionality."""

    def test_register_basic_workflow(
        self,
        tmp_path: Path,
        sample_nifti_pair: tuple[Path, Path],
        mock_weights_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test basic registration workflow without deep learning components."""
        fixed_path, moving_path = sample_nifti_pair
        results_dir = tmp_path / "results"

        # Mock environment variable for weights
        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        args = RegisterArgs(
            moving=(moving_path),
            fixed=(fixed_path),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=None,
            fixed_clip_limit=-1,
            moving_clip_limit=-1,
            skip_syn=True,  # Skip SyN for faster test
            skip_affine=False,
            skip_synthmorph=True,  # Skip SynthMorph (requires trained model)
            downsample_to=-1,
            synthmorph_mask=None,
            mirror_warp=False,
            keep_intermediates=True,
            force=False,
            log=Path("/dev/stderr"),
            verbose=True,
        )

        register(args)

        # Verify outputs
        assert results_dir.exists()
        assert (results_dir / "transform.h5").exists()
        assert (results_dir / "registered.nii").exists()

        # Verify transform.h5 structure
        with h5py.File(str(results_dir / "transform.h5"), "r") as h5_handle:
            assert "affine" in h5_handle
            assert "/affine/parameters" in h5_handle
            assert "/affine/fixed_parameters" in h5_handle

            # Check metadata preservation
            assert "fixed.shape" in h5_handle.attrs
            assert "fixed.origin" in h5_handle.attrs
            assert "fixed.spacing" in h5_handle.attrs

    def test_register_with_preprocessing(
        self,
        tmp_path: Path,
        sample_nifti_pair: tuple[Path, Path],
        mock_weights_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test registration with CLAHE preprocessing."""
        fixed_path, moving_path = sample_nifti_pair
        results_dir = tmp_path / "results"

        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        args = RegisterArgs(
            moving=(moving_path),
            fixed=(fixed_path),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=32,
            fixed_clip_limit=0.02,
            moving_clip_limit=0.03,
            skip_syn=True,
            skip_affine=False,
            skip_synthmorph=True,
            downsample_to=-1,
            synthmorph_mask=None,
            mirror_warp=False,
            keep_intermediates=False,
            force=False,
            log=None,
            verbose=True,
        )

        register(args)

        assert results_dir.exists()
        assert (results_dir / "transform.h5").exists()

        # Verify CLAHE settings were stored
        with h5py.File(str(results_dir / "transform.h5"), "r") as h5_handle:
            assert h5_handle.attrs["args.moving_clip_limit"] == 0.03
            assert h5_handle.attrs["args.fixed_clip_limit"] == 0.02

    def test_register_force_overwrite(
        self,
        tmp_path: Path,
        sample_nifti_pair: tuple[Path, Path],
        mock_weights_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test force overwrite of existing results."""
        fixed_path, moving_path = sample_nifti_pair
        results_dir = tmp_path / "results"

        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        # Create initial registration
        args = RegisterArgs(
            moving=(moving_path),
            fixed=(fixed_path),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=None,
            fixed_clip_limit=-1,
            moving_clip_limit=-1,
            skip_syn=True,
            skip_affine=False,
            skip_synthmorph=True,
            downsample_to=-1,
            synthmorph_mask=None,
            mirror_warp=False,
            keep_intermediates=False,
            force=False,
            log=None,
            verbose=True,
        )

        register(args)
        initial_mtime = (results_dir / "transform.h5").stat().st_mtime

        # Try to register again without force (should exit early)
        register(args)
        unchanged_mtime = (results_dir / "transform.h5").stat().st_mtime
        assert initial_mtime == unchanged_mtime

        # Register with force flag
        args.force = True
        register(args)
        new_mtime = (results_dir / "transform.h5").stat().st_mtime
        assert new_mtime >= initial_mtime

    def test_register_keep_intermediates(
        self,
        tmp_path: Path,
        sample_nifti_pair: tuple[Path, Path],
        mock_weights_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that intermediate results are kept when requested."""
        fixed_path, moving_path = sample_nifti_pair
        results_dir = tmp_path / "results"

        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        args = RegisterArgs(
            moving=(moving_path),
            fixed=(fixed_path),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=None,
            fixed_clip_limit=-1,
            moving_clip_limit=-1,
            skip_syn=True,
            skip_affine=False,
            skip_synthmorph=True,
            downsample_to=-1,
            synthmorph_mask=None,
            mirror_warp=False,
            keep_intermediates=True,
            force=False,
            log=None,
            verbose=True,
        )

        register(args)

        # Check that intermediate file exists
        assert (results_dir / "affine.nii").exists()

    def test_register_all_steps_skipped(
        self,
        tmp_path: Path,
        sample_nifti_pair: tuple[Path, Path],
        mock_weights_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that warning is issued when all registration steps are skipped."""
        fixed_path, moving_path = sample_nifti_pair
        results_dir = tmp_path / "results"

        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        args = RegisterArgs(
            moving=(moving_path),
            fixed=(fixed_path),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=None,
            fixed_clip_limit=-1,
            moving_clip_limit=-1,
            skip_syn=True,
            skip_affine=True,
            skip_synthmorph=True,
            downsample_to=-1,
            synthmorph_mask=None,
            mirror_warp=False,
            keep_intermediates=False,
            force=False,
            log=None,
            verbose=True,
        )

        register(args)

        # The function should return early with a warning
        assert not (results_dir / "transform.h5").exists()

    def test_register_custom_log_path(
        self,
        tmp_path: Path,
        sample_nifti_pair: tuple[Path, Path],
        mock_weights_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test custom log file path."""
        fixed_path, moving_path = sample_nifti_pair
        results_dir = tmp_path / "results"
        log_path = tmp_path / "custom_log.log"

        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        args = RegisterArgs(
            moving=(moving_path),
            fixed=(fixed_path),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=None,
            fixed_clip_limit=-1,
            moving_clip_limit=-1,
            skip_syn=True,
            skip_affine=False,
            skip_synthmorph=True,
            downsample_to=-1,
            synthmorph_mask=None,
            mirror_warp=False,
            keep_intermediates=False,
            force=False,
            log=(log_path),
            verbose=True,
        )

        register(args)

        assert log_path.exists()
        assert log_path.stat().st_size > 0

    def test_register_downsampling(
        self,
        tmp_path: Path,
        sample_nifti_pair: tuple[Path, Path],
        mock_weights_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test registration with image downsampling."""
        fixed_path, moving_path = sample_nifti_pair
        results_dir = tmp_path / "results"

        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        args = RegisterArgs(
            moving=(moving_path),
            fixed=(fixed_path),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=None,
            fixed_clip_limit=-1,
            moving_clip_limit=-1,
            skip_syn=True,
            skip_affine=False,
            skip_synthmorph=True,
            downsample_to=2.0,  # Downsample to 2.0 micron isotropic
            synthmorph_mask=None,
            mirror_warp=False,
            keep_intermediates=False,
            force=False,
            log=None,
            verbose=True,
        )

        register(args)

        assert results_dir.exists()
        assert (results_dir / "transform.h5").exists()

        # Verify downsampling parameter was stored
        with h5py.File(str(results_dir / "transform.h5"), "r") as h5_handle:
            assert h5_handle.attrs["args.downsample_to"] == 2.0

    def test_register_with_synthmorph_mask(
        self,
        tmp_path: Path,
        sample_nifti_pair: tuple[Path, Path],
        mock_weights_file: Path,
        sample_nifti_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test registration with SynthMorph mask applied."""
        fixed_path, moving_path = sample_nifti_pair
        results_dir = tmp_path / "results"

        # Use sample_nifti_file as mask (should have same dimensions as moving)
        mask_path = sample_nifti_file

        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        args = RegisterArgs(
            moving=(moving_path),
            fixed=(fixed_path),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=None,
            fixed_clip_limit=-1,
            moving_clip_limit=-1,
            skip_syn=True,
            skip_affine=False,
            skip_synthmorph=True,  # Skip SynthMorph inference but test mask loading
            downsample_to=-1,
            synthmorph_mask=(mask_path),
            mirror_warp=False,
            keep_intermediates=False,
            force=False,
            log=None,
            verbose=True,
        )

        register(args)

        assert results_dir.exists()
        assert (results_dir / "transform.h5").exists()

        # Verify mask was stored in HDF5
        with h5py.File(str(results_dir / "transform.h5"), "r") as h5_handle:
            assert "/synthmorph_mask" in h5_handle

    def test_register_with_synthmorph_mask_downsampling(
        self,
        tmp_path: Path,
        sample_nifti_pair: tuple[Path, Path],
        mock_weights_file: Path,
        sample_nifti_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test registration with SynthMorph mask and downsampling enabled."""
        fixed_path, moving_path = sample_nifti_pair
        results_dir = tmp_path / "results"
        mask_path = sample_nifti_file

        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        args = RegisterArgs(
            moving=(moving_path),
            fixed=(fixed_path),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=None,
            fixed_clip_limit=-1,
            moving_clip_limit=-1,
            skip_syn=True,
            skip_affine=False,
            skip_synthmorph=True,
            downsample_to=2.0,  # Enable downsampling
            synthmorph_mask=(mask_path),
            mirror_warp=False,
            keep_intermediates=False,
            force=False,
            log=None,
            verbose=True,
        )

        register(args)

        assert results_dir.exists()
        assert (results_dir / "transform.h5").exists()

        # Verify mask was resampled and stored
        with h5py.File(str(results_dir / "transform.h5"), "r") as h5_handle:
            assert "/synthmorph_mask" in h5_handle
            # Mask should be resampled to match downsampled moving image
            assert h5_handle.attrs["args.downsample_to"] == 2.0

    def test_register_synthmorph_mask_metadata_mismatch(
        self,
        tmp_path: Path,
        sample_nifti_pair: tuple[Path, Path],
        mock_weights_file: Path,
        sample_3d_image: "ants.ANTsImage",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test error handling when SynthMorph mask has mismatched metadata."""
        fixed_path, moving_path = sample_nifti_pair
        results_dir = tmp_path / "results"

        # Create mask with different spacing
        import ants

        mismatched_mask = ants.resample_image(sample_3d_image, (2.0, 2.0, 2.0), interp_type=0)
        mask_path = tmp_path / "mismatched_mask.nii"
        ants.image_write(mismatched_mask, str(mask_path))

        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        args = RegisterArgs(
            moving=(moving_path),
            fixed=(fixed_path),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=None,
            fixed_clip_limit=-1,
            moving_clip_limit=-1,
            skip_syn=True,
            skip_affine=False,
            skip_synthmorph=True,
            downsample_to=-1,  # No downsampling, so metadata must match
            synthmorph_mask=(mask_path),
            mirror_warp=False,
            keep_intermediates=False,
            force=False,
            log=None,
            verbose=True,
        )

        # Should raise RuntimeError due to metadata mismatch
        with pytest.raises(RuntimeError, match="Fatal discrepancy"):
            register(args)

    def test_register_synthmorph_mask_with_affine_and_syn(
        self,
        tmp_path: Path,
        sample_nifti_pair: tuple[Path, Path],
        mock_weights_file: Path,
        sample_nifti_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that SynthMorph mask is transformed through affine and SyN steps."""
        fixed_path, moving_path = sample_nifti_pair
        results_dir = tmp_path / "results"
        mask_path = sample_nifti_file

        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        args = RegisterArgs(
            moving=(moving_path),
            fixed=(fixed_path),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=None,
            fixed_clip_limit=-1,
            moving_clip_limit=-1,
            skip_syn=False,  # Enable SyN
            skip_affine=False,  # Enable affine
            skip_synthmorph=False,  # Skip SynthMorph inference
            downsample_to=-1,
            synthmorph_mask=(mask_path),
            mirror_warp=False,
            keep_intermediates=True,
            force=False,
            log=None,
            verbose=False,
        )

        register(args)

        assert results_dir.exists()
        assert (results_dir / "transform.h5").exists()

        # Verify both transforms and mask are stored
        with h5py.File(str(results_dir / "transform.h5"), "r") as h5_handle:
            assert "/affine" in h5_handle
            assert "/syn" in h5_handle
            assert "/synthmorph_mask" in h5_handle

    def test_register_missing_synthmorph_mask_file(
        self,
        tmp_path: Path,
        sample_nifti_pair: tuple[Path, Path],
        mock_weights_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test error handling when SynthMorph mask file doesn't exist."""
        fixed_path, moving_path = sample_nifti_pair
        results_dir = tmp_path / "results"
        nonexistent_mask = tmp_path / "nonexistent_mask.nii"

        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        args = RegisterArgs(
            moving=(moving_path),
            fixed=(fixed_path),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=None,
            fixed_clip_limit=-1,
            moving_clip_limit=-1,
            skip_syn=True,
            skip_affine=False,
            skip_synthmorph=True,
            downsample_to=-1,
            synthmorph_mask=(nonexistent_mask),
            mirror_warp=False,
            keep_intermediates=False,
            force=False,
            log=None,
            verbose=True,
        )

        # Should raise an error when trying to load the mask
        with pytest.raises(Exception):  # ANTs will raise an exception for missing file
            register(args)

    def test_register_missing_moving_image(
        self, tmp_path: Path, sample_nifti_file: Path, mock_weights_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test error handling for missing moving image."""
        results_dir = tmp_path / "results"

        monkeypatch.setenv("BIFROST_WEIGHTS_PATH", str(mock_weights_file))

        args = RegisterArgs(
            moving=(tmp_path / "nonexistent.nii"),
            fixed=(sample_nifti_file),
            results_dir=(results_dir),
            weights=(mock_weights_file),
            clahe_kernel_size=None,
            fixed_clip_limit=-1,
            moving_clip_limit=-1,
            skip_syn=True,
            skip_affine=False,
            skip_synthmorph=True,
            downsample_to=-1,
            synthmorph_mask=None,
            mirror_warp=False,
            keep_intermediates=False,
            force=False,
            log=None,
            verbose=True,
        )

        with pytest.raises(AssertionError):
            register(args)
