from pathlib import Path
import random
import shutil
import cv2
import numpy as np
import albumentations as A


# ============================================================
# SETTINGS
# ============================================================

# Your 36 letter/number dataset
SOURCE_DIR = Path(r"D:\data")

# Final combined augmented dataset.
# Your 195 ISL classes can already be here.
DEST_DIR = Path(r"D:\aug_indian_dataset")

# For EACH letter/number class:
#   100 original copies
#   500 augmented images
#   -------------------
#   600 total images
ORIGINAL_TARGET = 100
AUGMENTED_TARGET = 500
TOTAL_TARGET = ORIGINAL_TARGET + AUGMENTED_TARGET

SEED = 42

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
# Included:
#   - left/right rotation
#   - tilt
#   - shear
#   - small translation
#   - zoom in
#   - zoom out
#   - perspective
#   - brightness
#   - contrast
#   - gamma
#   - slight blur
#   - slight motion blur
#   - slight noise
#   - JPEG compression
#
# NO horizontal flip:
# Left/right orientation can matter for hand signs.
# ============================================================

AUGMENTER = A.Compose([

    # Rotation + tilt + zoom + shear + translation
    A.Affine(
        scale=(0.88, 1.12),
        translate_percent=(-0.06, 0.06),
        rotate=(-18, 18),
        shear=(-8, 8),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.85
    ),

    # Small perspective change
    A.Perspective(
        scale=(0.01, 0.05),
        keep_size=True,
        pad_mode=cv2.BORDER_REFLECT_101,
        p=0.20
    ),

    # Lighting changes
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.45
    ),

    A.RandomGamma(
        gamma_limit=(85, 115),
        p=0.25
    ),

    # Slight blur
    A.OneOf([
        A.GaussianBlur(
            blur_limit=(3, 5)
        ),
        A.MotionBlur(
            blur_limit=5
        )
    ], p=0.12),

    # Small realistic sensor noise
    A.GaussNoise(
        std_range=(0.005, 0.025),
        p=0.10
    ),

    # Slight compression
    A.ImageCompression(
        quality_range=(80, 100),
        p=0.10
    )
])


# ============================================================
# HELPERS
# ============================================================

def get_images(folder):
    return sorted([
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def save_jpg(path, image):
    return cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )


def unique_path(folder, stem, extension=".jpg"):
    """
    Create a filename that does not overwrite another image.
    """

    path = folder / f"{stem}{extension}"

    if not path.exists():
        return path

    counter = 1

    while True:
        path = folder / f"{stem}_{counter}{extension}"

        if not path.exists():
            return path

        counter += 1


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("LETTER + NUMBER DATA AUGMENTATION")
    print("=" * 80)

    print("\nSource:")
    print(SOURCE_DIR)

    print("\nFinal dataset:")
    print(DEST_DIR)

    print(
        f"\nPer class: "
        f"{ORIGINAL_TARGET} original + "
        f"{AUGMENTED_TARGET} augmented = "
        f"{TOTAL_TARGET}"
    )


    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not SOURCE_DIR.exists():

        print("\nERROR: Source folder does not exist:")
        print(SOURCE_DIR)
        return


    # --------------------------------------------------------
    # Create destination
    # --------------------------------------------------------

    DEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Find classes
    # --------------------------------------------------------

    class_folders = sorted([
        p for p in SOURCE_DIR.iterdir()
        if p.is_dir()
    ])


    if not class_folders:

        print("\nERROR: No class folders found.")
        return


    print(
        f"\nClasses found: {len(class_folders)}"
    )

    if len(class_folders) != 36:

        print(
            "\nWARNING: Expected 36 letter/number classes."
        )

        print(
            "The script will process the folders that "
            "are actually present."
        )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_original = 0
    total_augmented = 0
    total_final = 0


    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for class_index, class_folder in enumerate(
        class_folders,
        start=1
    ):

        class_name = class_folder.name

        source_images = get_images(
            class_folder
        )


        if not source_images:

            print(
                f"[SKIP] {class_name}: no images"
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


        print(
            f"\n[{class_index:02d}/{len(class_folders):02d}] "
            f"{class_name}"
        )

        print(
            f"    Source images: {len(source_images)}"
        )


        # ====================================================
        # PART 1: 100 ORIGINAL IMAGES
        # ====================================================
        #
        # If there are >= 100 source images:
        #     use 100 of them.
        #
        # If there are < 100:
        #     repeat source images until we reach 100.
        #
        # These are NOT augmented.
        # ====================================================

        original_count = 0

        for i in range(ORIGINAL_TARGET):

            source_path = source_images[
                i % len(source_images)
            ]

            image = cv2.imread(
                str(source_path)
            )

            if image is None:

                print(
                    f"    Warning: could not read "
                    f"{source_path.name}"
                )

                continue


            output_path = unique_path(
                destination_class,
                f"{class_name}_original_{i+1:04d}"
            )


            if save_jpg(
                output_path,
                image
            ):

                original_count += 1
                total_original += 1


        # ====================================================
        # PART 2: 500 AUGMENTED IMAGES
        # ====================================================

        augmented_count = 0

        source_index = 0

        while augmented_count < AUGMENTED_TARGET:

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

            except Exception as error:

                print(
                    f"    Augmentation error: {error}"
                )

                continue


            output_path = unique_path(
                destination_class,
                f"{class_name}_aug_{augmented_count+1:04d}"
            )


            if save_jpg(
                output_path,
                augmented
            ):

                augmented_count += 1
                total_augmented += 1


        final_count = (
            original_count +
            augmented_count
        )

        total_final += final_count


        print(
            f"    Original copied : "
            f"{original_count}"
        )

        print(
            f"    Augmented       : "
            f"{augmented_count}"
        )

        print(
            f"    Final           : "
            f"{final_count}"
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("LETTER/NUMBER AUGMENTATION COMPLETE")
    print("=" * 80)

    print(
        f"Classes processed : {len(class_folders)}"
    )

    print(
        f"Original images   : {total_original:,}"
    )

    print(
        f"Augmented images  : {total_augmented:,}"
    )

    print(
        f"Final images      : {total_final:,}"
    )

    print(
        f"Target per class  : {TOTAL_TARGET}"
    )

    print(
        f"\nOutput:"
    )

    print(
        DEST_DIR
    )

    print("\nExpected for 36 classes:")

    print(
        f"36 × {TOTAL_TARGET} = "
        f"{36 * TOTAL_TARGET:,} images"
    )

    print("\nIMPORTANT:")
    print(
        "Existing folders in D:\\aug_indian_dataset "
        "are NOT deleted."
    )

    print(
        "The 1-9/A-Z folders are added to the same "
        "final dataset."
    )

    print(
        "\nHorizontal flip is intentionally disabled."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
