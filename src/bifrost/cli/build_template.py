"""Logic for template building.

Execute using the 'bifrost' executable installed by setuptools
"""

import logging
import shutil
from pathlib import Path

import ants
import numpy as np
import scipy
import scipy.ndimage
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_triangle as triangle
from sklearn.preprocessing import quantile_transform

from bifrost.cli.bifrost import BuildTemplateArgs
from bifrost.io import guarded_ants_image_read
from bifrost.util import setup_cli_logger, sha256, update_image_array


def build_template(args: BuildTemplateArgs) -> None:
    # ========================================================================== #
    #                      CONFIGURE LOGGER                                      #
    # ========================================================================== #

    logger = setup_cli_logger(__name__, args.verbose, args.log)

    try:
        # ========================================================================== #
        #                              PATH LOGIC                                    #
        # ========================================================================== #

        input_paths: list[Path] = []

        for input_path_str in args.input:
            input_path = Path(input_path_str)
            assert input_path.exists()

            if input_path.is_dir():
                input_paths.extend(input_path.iterdir())
            else:
                input_paths.append(input_path)

        logger.debug("Parsed input files: %s", input_paths)

        reference_image_path: Path | None = None
        if args.reference_image is not None:
            reference_image_path = Path(args.reference_image)
            assert reference_image_path.exists()

        if args.output.exists():
            if args.force:
                logger.info("Cleaning existing results directory")
                shutil.rmtree(args.output, ignore_errors=True)
            elif args.preemptible:
                logger.info("Resuming existing work")
            else:
                logger.warning(
                    "Results directory already exists. Run again with --force to override or --preemptible to resume"
                )
                return

        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "preprocessed").mkdir(exist_ok=True)
        (args.output / "templates").mkdir(exist_ok=True)
        (args.output / "scratch").mkdir(exist_ok=True)

        # ========================================================================== #
        #                      CONFIGURE LOG FILE HANDLER                            #
        # ========================================================================== #

        log_path = args.output / "build_template.log" if args.log is None else Path(args.log)

        logger.info("Writings logs to %s", log_path)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        logger.debug("Parsed args: %s", args)

        # ==================================================================== #
        #                             PREPROCESSING                            #
        # ==================================================================== #

        logger.info(
            "Preprocessing images: %s",
            input_paths,
        )

        for input_path in input_paths:
            name = input_path.stem
            output_path = args.output / "preprocessed" / f"{sha256(str(input_path).encode())}_{name}.nii"

            if not output_path.exists():
                logger.info("Preprocessing %s", input_path)
                preprocess(args, input_path, output_path)
            else:
                logger.info("Cached result found for %s preprocessing", name)

        # ========================================================================== #
        #                                  AFFINE                                    #
        # ========================================================================== #

        initial_image = reference_image_path if reference_image_path is not None else input_paths[0]

        logger.info("Starting affine step 0")
        alignment_iteration(
            args,
            moving_dir=args.output / "preprocessed",
            step_name="affine_0",
            fixed_path=initial_image,
            type_of_transform="Affine",
            transform_avg=False,
            mirror=args.mirror,
        )

        for affine_step in range(1, args.affine_steps):
            logger.info("Starting affine step %s", affine_step)

            alignment_iteration(
                args,
                moving_dir=args.output / "scratch" / f"affine_{affine_step - 1}",
                step_name=f"affine_{affine_step}",
                fixed_path=args.output / "templates" / f"affine_{affine_step - 1}.nii",
                type_of_transform="Affine",
                transform_avg=False,
                mirror=False,
            )

        # ========================================================================== #
        #                                    SyN                                     #
        # ========================================================================== #

        logger.info("Starting SyN step 0")

        alignment_iteration(
            args,
            moving_dir=args.output / "scratch" / f"affine_{args.affine_steps - 1}",
            step_name="syn_0",
            fixed_path=args.output / "templates" / f"affine_{args.affine_steps - 1}.nii",
            type_of_transform="SyN",
            transform_avg=True,
            mirror=False,
        )

        for syn_step in range(1, args.syn_steps):
            logger.info("Starting SyN step %s", syn_step)

            alignment_iteration(
                args,
                moving_dir=args.output / "scratch" / f"syn_{syn_step - 1}",
                step_name=f"syn_{syn_step}",
                fixed_path=args.output / "templates" / f"syn_{syn_step - 1}.nii",
                type_of_transform="SyN",
                transform_avg=True,
                mirror=False,
            )

        logger.info("Cleaning up")

        final_template = args.output / "template.nii"
        shutil.move(
            str(args.output / "templates" / f"syn_{args.syn_steps - 1}.nii"),
            str(final_template),
        )

        if args.keep_intermediates:
            # add back symlink for the final result which was moved
            symlink_path = args.output / "templates" / f"syn_{args.syn_steps}.nii"
            symlink_path.symlink_to(final_template)
        else:
            shutil.rmtree(args.output / "preprocessed", ignore_errors=True)
            shutil.rmtree(args.output / "templates", ignore_errors=True)

        logger.info("Template generation complete")

    except:
        logger.exception("Caught general exception")
        raise

    finally:
        logger.info("Exiting, cleaning scratch")
        shutil.rmtree(args.output / "scratch", ignore_errors=True)


def preprocess(args: BuildTemplateArgs, input_path: Path, output_path: Path) -> None:
    """Runs all preprocessing."""
    image = guarded_ants_image_read(str(input_path))

    if args.preprocessing is None:
        shutil.copy(input_path, output_path)
        return

    if "legacy" in args.preprocessing:
        image = __legacy_preprocess(image)

    if "CLAHE" in args.preprocessing:
        image -= image.min()
        image /= image.max()

        image = update_image_array(image, equalize_adapthist(image.numpy(), kernel_size=64, clip_limit=0.03))

    ants.image_write(image, str(output_path))


def __legacy_preprocess(image: ants.ANTsImage) -> ants.ANTsImage:
    """Legacy preprocessing."""
    image_arr = image.numpy()

    # Blur brain and mask small values
    image_copy = image_arr.copy().astype("float32")
    image_copy = scipy.ndimage.gaussian_filter(image_copy, sigma=10)
    threshold = triangle(image_copy)
    image_copy[np.where(image_copy < threshold / 2)] = 0.0

    # Remove blobs outside contiguous brain
    labels, _label_nb = scipy.ndimage.label(image_copy)
    image_label = (np.bincount(labels.flatten())[1:].argmax()) + 1
    image_copy = image_arr.copy().astype("float32")
    image_copy[np.where(labels != image_label)] = np.nan

    # Perform quantile normalization
    image_copy = quantile_transform(image_copy.flatten().reshape(-1, 1), n_quantiles=500, random_state=0)
    image_copy = image_copy.reshape(image_arr.shape)

    return update_image_array(image, np.nan_to_num(image_copy))


def generate_template(args: BuildTemplateArgs, step_name: str, output_path: Path, transform_avg: bool) -> None:
    """Generates template from registration results.

    Depending on the experiment type this either averages the images directly or 'averages' their transformations
    """
    logger = logging.getLogger(__name__)
    __retries = 0

    assert output_path.suffix == ".nii"

    while True:
        try:
            input_dir = args.output / "scratch" / step_name

            # 'average' transformations
            #
            # NOTE: a shortcoming of this method is that the affine transform is assumed to be nearly the identity
            #  ie, it is ignored entirely
            # there doesn't seem to be a well defined method to average affine transformations
            # however, much work has been done on averaging quaternions
            # it is also unclear whether averaging the 'diffeomorphic' warp is well-defined
            # it is, at least, a crude hack as implemented (https://github.com/ANTsX/ANTsPy/issues/125)
            #
            if transform_avg:
                transform_dir = args.output / "scratch" / f"{step_name}_transform"

                if (transform_dir / "transform.nii").exists():
                    logger.info("%s: found existing inverse average transform", step_name)
                    avg_img = __average_images(input_dir / "*.nii")
                else:
                    transform_dir.mkdir(parents=True, exist_ok=True)

                    avg_img = __average_images(input_dir / "*.nii")
                    avg_transform = __average_images(input_dir / "*.nii.gz")

                    # this could only ever be construed as an inverse if you squint, a lot
                    inv_avg_transform = avg_transform * -1 * args.gradient_step

                    ants.image_write(inv_avg_transform, str(transform_dir / "transform.nii"))

                template = ants.apply_transforms(avg_img, avg_img, str(transform_dir / "transform.nii"))

                ants.image_write(template, str(output_path))

            # average images directly
            else:
                template = __average_images(input_dir / "*.nii")
                ants.image_write(template, str(output_path))

            break
        except Exception as exc:
            if __retries < 2:
                logger.exception("Caught exception, retrying")

                __clean_template(output_path)
                __retries += 1
            else:
                logger.exception("Caught exception, max retries reached")
                raise exc


def alignment_iteration(
    args: BuildTemplateArgs,
    moving_dir: Path,
    step_name: str,
    fixed_path: Path,
    type_of_transform: str,
    transform_avg: bool,
    mirror: bool,
) -> None:
    logger = logging.getLogger(__name__)
    __retries = 0

    step_dir = args.output / "scratch" / step_name
    step_dir.mkdir(parents=True, exist_ok=True)

    template_path = args.output / "templates" / f"{step_name}.nii"
    if template_path.exists():
        logger.info(f"{step_name} template already exists")
        return

    fixed = ants.image_read(str(fixed_path))

    for input_path in moving_dir.glob("*.nii"):
        input_name = None

        while True:
            try:
                input_name = input_path.stem

                if __step_output_exists(input_name, step_dir, transform_avg, False):
                    logger.info("%s: found cached result for %s ", step_name, input_name)
                else:
                    logger.info("%s: processing %s", step_name, input_name)

                    moving = ants.image_read(str(input_path))

                    registration = ants.registration(
                        fixed,
                        moving,
                        type_of_transform=type_of_transform,
                        verbose=args.verbose,
                        outprefix=str(step_dir / input_name),
                    )

                    __write_step_output(
                        registration,
                        input_name,
                        step_dir,
                        write_transform=transform_avg,
                        mirror=False,
                    )

                if mirror:
                    if __step_output_exists(input_name, step_dir, transform_avg, True):
                        logger.info(
                            "%s: found existing work for input %s mirror",
                            step_name,
                            input_name,
                        )
                    else:
                        logger.info("%s: processing input %s mirror", step_name, input_name)

                        moving_mirror = update_image_array(moving, moving[::-1])

                        registration_mirror = ants.registration(
                            fixed,
                            moving_mirror,
                            type_of_transform=type_of_transform,
                            verbose=args.verbose,
                            outprefix=str(step_dir / f"{input_name}_m"),
                        )

                        __write_step_output(
                            registration_mirror,
                            input_name,
                            step_dir,
                            write_transform=transform_avg,
                            mirror=True,
                        )
                break
            except Exception as exc:
                if __retries < 2:
                    logger.exception("Caught exception processing input %s, retrying", input_name)

                    __clean_step_output(input_name, step_dir, mirror)
                    __retries += 1
                else:
                    logger.exception(
                        "Caught exception processing input %s, max retries reached",
                        input_name,
                    )
                    raise exc

    logger.info("Generating new template for  %s", step_name)
    generate_template(
        args,
        step_name=step_name,
        output_path=template_path,
        transform_avg=transform_avg,
    )

    logger.info("Finished %s", step_name)


def __average_images(pattern: Path) -> ants.ANTsImage:
    img_paths = list(pattern.parent.glob(pattern.name))

    img_0 = ants.image_read(str(img_paths[0]))
    avg_img = img_0.numpy() / len(img_paths)

    for img_path in img_paths[1:]:
        avg_img += ants.image_read(str(img_path)).numpy() / len(img_paths)

    return update_image_array(img_0, avg_img)


def __write_step_output(
    registration: dict, input_name: str, step_dir: Path, write_transform: bool, mirror: bool
) -> None:
    suffix = "_m" if mirror else ""

    ants.image_write(
        registration["warpedmovout"],
        str(step_dir / f"{input_name}{suffix}.nii"),
    )

    if write_transform:
        shutil.copy(
            registration["fwdtransforms"][0],
            str(step_dir / f"{input_name}{suffix}_t.nii.gz"),
        )


def __step_output_exists(input_name: str, step_dir: Path, write_transform: bool, mirror: bool) -> bool:
    suffix = "_m" if mirror else ""

    if not (step_dir / f"{input_name}{suffix}.nii").exists():
        return False

    return not (write_transform and not (step_dir / f"{input_name}{suffix}_t.nii.gz").exists())


def __clean_step_output(input_name: str, step_dir: Path, mirror: bool) -> None:
    suffix = "_m" if mirror else ""

    (step_dir / f"{input_name}{suffix}.nii").unlink(missing_ok=True)
    (step_dir / f"{input_name}{suffix}_t.nii.gz").unlink(missing_ok=True)


def __clean_template(template_path: Path) -> None:
    template_path.unlink(missing_ok=True)
