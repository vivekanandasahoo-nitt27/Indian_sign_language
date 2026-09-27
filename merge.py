from pathlib import Path
import shutil
import re
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = Path(r"D:\indian_sign_test")

# We create a NEW merged dataset so the current dataset is safe.
DEST_DIR = Path(r"D:\indian_sign_merged_test")


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# CLASS NAME RULE
# ============================================================

def get_base_class(folder_name):
    """
    Remove ONLY trailing digits from the folder name.

    Examples:
        Again       -> Again
        Again1      -> Again
        Again2      -> Again

        Arrest      -> Arrest
        Arrest1     -> Arrest
        Arrest2     -> Arrest

        Ascend      -> Ascend
        Ascend1     -> Ascend
        Ascend2     -> Ascend

        Bad1        -> Bad
        Bad2        -> Bad

    IMPORTANT:
    Digits are removed ONLY when they are at the END.
    """

    base_name = re.sub(
        r"\d+$",
        "",
        folder_name
    ).strip()

    return base_name if base_name else folder_name


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("INDIAN SIGN LANGUAGE - MERGE NUMBERED CLASSES")
    print("=" * 80)

    print("\nSource:")
    print(SOURCE_DIR)

    print("\nMerged destination:")
    print(DEST_DIR)


    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not SOURCE_DIR.exists():

        print("\nERROR: Source dataset does not exist.")

        print(
            f"Expected folder:\n{SOURCE_DIR}"
        )

        return


    # --------------------------------------------------------
    # Prevent accidental overwrite
    # --------------------------------------------------------

    if DEST_DIR.exists():

        print(
            "\nERROR: Destination already exists:"
        )

        print(DEST_DIR)

        print(
            "\nDelete/rename that folder first if you want "
            "to run the merge again."
        )

        return


    DEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Find class folders
    # --------------------------------------------------------

    class_folders = [
        p
        for p in SOURCE_DIR.iterdir()
        if p.is_dir()
    ]


    print(
        f"\nOriginal class folders: "
        f"{len(class_folders):,}"
    )


    # --------------------------------------------------------
    # Group folders by base class
    # --------------------------------------------------------

    grouped = defaultdict(list)

    for folder in class_folders:

        base_class = get_base_class(
            folder.name
        )

        grouped[base_class].append(
            folder
        )


    print(
        f"Merged class count: "
        f"{len(grouped):,}"
    )


    # --------------------------------------------------------
    # Show merge plan
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("MERGE PLAN")
    print("-" * 80)

    for base_class in sorted(grouped):

        folders = sorted(
            grouped[base_class],
            key=lambda x: x.name.lower()
        )

        if len(folders) > 1:

            names = ", ".join(
                folder.name
                for folder in folders
            )

            print(
                f"{names}  -->  {base_class}"
            )


    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    total_images = 0
    total_classes = 0

    print("\n" + "-" * 80)
    print("COPYING IMAGES")
    print("-" * 80)


    for base_class in sorted(grouped):

        destination_class = (
            DEST_DIR /
            base_class
        )

        destination_class.mkdir(
            parents=True,
            exist_ok=True
        )

        total_classes += 1

        class_count = 0

        for source_class_folder in sorted(
            grouped[base_class],
            key=lambda x: x.name.lower()
        ):

            image_files = [
                p
                for p in source_class_folder.iterdir()
                if (
                    p.is_file()
                    and
                    p.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]

            for image_file in image_files:

                destination_file = (
                    destination_class /
                    image_file.name
                )


                # Filename collision protection
                if destination_file.exists():

                    stem = image_file.stem
                    suffix = image_file.suffix

                    counter = 1

                    while True:

                        new_name = (
                            f"{stem}_{counter}"
                            f"{suffix}"
                        )

                        destination_file = (
                            destination_class /
                            new_name
                        )

                        if not destination_file.exists():
                            break

                        counter += 1


                shutil.copy2(
                    image_file,
                    destination_file
                )

                class_count += 1
                total_images += 1


        print(
            f"{base_class:<40}"
            f"{class_count:>7} images"
        )


    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MERGING COMPLETE")
    print("=" * 80)

    print(
        f"Original folders : {len(class_folders):,}"
    )

    print(
        f"Final classes    : {total_classes:,}"
    )

    print(
        f"Total images     : {total_images:,}"
    )

    print(
        f"\nFinal dataset:"
    )

    print(
        DEST_DIR
    )

    print("\nExample:")

    print(
        "  Again + Again1 + Again2"
    )

    print(
        "        -> Again/"
    )

    print(
        "  Arrest + Arrest1"
    )

    print(
        "        -> Arrest/"
    )

    print(
        "  Ascend + Ascend1 + Ascend2"
    )

    print(
        "        -> Ascend/"
    )

    print("\nIMPORTANT:")
    print(
        "The original D:\\indian_sign_language dataset "
        "was NOT modified."
    )

    print(
        "The merged dataset is in "
        "D:\\indian_sign_language_merged"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
