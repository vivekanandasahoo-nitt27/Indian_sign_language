from pathlib import Path
import random
import shutil
import cv2
import numpy as np
import albumentations as A


# ============================================================
# SETTINGS
# ============================================================

# Input = the merged dataset from the previous step
SOURCE_DIR = Path(r"D:\indian_sign_language_merged")

# New augmented dataset
DEST_DIR = Path(r"D:\aug_indian_dataset")

# Target number of images PER CLASS in the final dataset
TARGET_IMAGES_PER_CLASS = 600

# Reproducibility
SEED = 42

# Supported images
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# AUGMENTATION PIPELINE
# ============================================================
#
# IMPORTANT FOR SIGN LANGUAGE:
# We DO NOT use horizontal flipping because left/right
# handedness can be meaningful.
#
# The pipeline includes:
#   - small left/right rotation
#   - tilt / affine transformation
#   - slight perspective
#   - zoom in
#   - zoom out
#   - brightness adjustment
#   - contrast adjustment
#   - gamma adjustment
#   - slight blur
#   - slight Gaussian noise
#   - JPEG compression
#
# Each generated image receives a random combination.
# ============================================================

AUGMENTER = A.Compose([
    A.Affine(
        scale=(0.90, 1.10),
        translate_percent=(-0.06, 0.06),
        rotate=(-18, 18),
        shear=(-8, 8),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.85
    ),

    A.Perspective(
        scale=(0.015, 0.055),
        keep_size=True,
        pad_mode=cv2.BORDER_REFLECT_101,
        p=0.20
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.45
    ),

    A.RandomGamma(
        gamma_limit=(85, 115),
        p=0.25
    ),

    A.OneOf([
        A.GaussianBlur(
            blur_limit=(3, 5)
        ),
        A.MotionBlur(
            blur_limit=5
        ),
    ], p=0.15),

    A.GaussNoise(
        std_range=(0.01, 0.04),
        p=0.15
    ),

    A.ImageCompression(
        quality_range=(75, 100),
        p=0.15
    ),
])


# ============================================================
# HELPERS
# ============================================================

def get_images(folder):
    """Return all image files directly inside a class folder."""

    return [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def save_image(path, image):
    """Save an image with high JPEG quality."""

    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        return cv2.imwrite(
            str(path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

    return cv2.imwrite(
        str(path),
        image
    )


def make_augmented_name(
    class_name,
    index,
    source_index
):
    return (
        f"{class_name}_aug_"
        f"{index:06d}_src{source_index:05d}.jpg"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("INDIAN SIGN LANGUAGE - DATA AUGMENTATION")
    print("=" * 80)

    print("\nSource:")
    print(SOURCE_DIR)

    print("\nDestination:")
    print(DEST_DIR)

    print(
        f"\nTarget images per class: "
        f"{TARGET_IMAGES_PER_CLASS}"
    )

    print(
        "\nHorizontal flip: DISABLED "
        "(important for handedness)"
    )


    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not SOURCE_DIR.exists():

        raise FileNotFoundError(
            f"\nSource dataset not found:\n{SOURCE_DIR}"
        )


    # --------------------------------------------------------
    # Check destination
    # --------------------------------------------------------

    if DEST_DIR.exists():

        existing_items = list(
            DEST_DIR.iterdir()
        )

        if existing_items:

            print(
                "\nERROR: Destination already contains data:"
            )

            print(DEST_DIR)

            print(
                "\nDelete or rename this folder before "
                "running the script again."
            )

            return

    else:

        DEST_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


    # --------------------------------------------------------
    # Find class folders
    # --------------------------------------------------------

    class_folders = sorted([
        p for p in SOURCE_DIR.iterdir()
        if p.is_dir()
    ])


    if not class_folders:

        raise RuntimeError(
            "No class folders found in source dataset."
        )


    print(
        f"\nClasses found: {len(class_folders):,}"
    )


    # --------------------------------------------------------
    # Process every class
    # --------------------------------------------------------

    total_original = 0
    total_augmented = 0
    total_final = 0

    print("\n" + "-" * 80)
    print("PROCESSING CLASSES")
    print("-" * 80)


    for class_number, class_folder in enumerate(
        class_folders,
        start=1
    ):

        class_name = class_folder.name

        source_images = get_images(
            class_folder
        )

        if not source_images:

            print(
                f"[SKIP] {class_name}: "
                "no images"
            )

            continue


        destination_class = (
            DEST_DIR /
            class_name
        )

        destination_class.mkdir(
            parents=True,
            exist_ok=True
        )


        # ----------------------------------------------------
        # COPY ORIGINAL IMAGES FIRST
        # ----------------------------------------------------

        for source_image in source_images:

            image = cv2.imread(
                str(source_image)
            )

            if image is None:
                continue

            destination_name = (
                f"{class_name}_original_"
                f"{len(list(destination_class.glob('*.jpg'))):06d}.jpg"
            )

            destination_path = (
                destination_class /
                destination_name
            )

            if save_image(
                destination_path,
                image
            ):

                total_original += 1


        # Count originals actually copied
        current_count = len(
            get_images(destination_class)
        )


        # ----------------------------------------------------
        # GENERATE AUGMENTED IMAGES UNTIL 600
        # ----------------------------------------------------

        augmented_count = 0

        source_index = 0

        while current_count < TARGET_IMAGES_PER_CLASS:

            source_path = source_images[
                source_index % len(source_images)
            ]

            source_index += 1


            image = cv2.imread(
                str(source_path)
            )

            if image is None:
                continue


            try:

                augmented = AUGMENTER(
                    image=image
                )["image"]

            except Exception as e:

                print(
                    f"\nAugmentation error in "
                    f"{class_name}: {e}"
                )

                continue


            output_name = make_augmented_name(
                class_name,
                augmented_count,
                source_index
            )

            output_path = (
                destination_class /
                output_name
            )


            if save_image(
                output_path,
                augmented
            ):

                augmented_count += 1
                current_count += 1
                total_augmented += 1


        total_final += current_count


        print(
            f"[{class_number:>3}/"
            f"{len(class_folders):<3}] "
            f"{class_name:<35} "
            f"original={len(source_images):>4}  "
            f"augmented={augmented_count:>4}  "
            f"final={current_count:>4}"
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("AUGMENTATION COMPLETE")
    print("=" * 80)

    print(
        f"Classes processed : "
        f"{len(class_folders):,}"
    )

    print(
        f"Original images   : "
        f"{total_original:,}"
    )

    print(
        f"New augmented     : "
        f"{total_augmented:,}"
    )

    print(
        f"Final images      : "
        f"{total_final:,}"
    )

    print(
        f"Target per class  : "
        f"{TARGET_IMAGES_PER_CLASS}"
    )

    print(
        f"\nFinal dataset:"
    )

    print(
        DEST_DIR
    )

    print("\nEach class should now contain")
    print(
        f"approximately/exactly "
        f"{TARGET_IMAGES_PER_CLASS} images."
    )

    print("\nAugmentations used:")
    print("  - rotation left/right")
    print("  - tilt / shear")
    print("  - small translation")
    print("  - zoom in")
    print("  - zoom out")
    print("  - perspective")
    print("  - brightness")
    print("  - contrast")
    print("  - gamma")
    print("  - slight blur")
    print("  - slight motion blur")
    print("  - slight Gaussian noise")
    print("  - JPEG compression")
    print("  - NO horizontal flip")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
