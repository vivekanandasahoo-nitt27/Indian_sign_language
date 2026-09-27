from pathlib import Path
import shutil
from collections import defaultdict

# Source shown in your screenshot:
SOURCE_DIR = Path(r"D:\Dataset_test")

# Final dataset:
DEST_DIR = Path(r"D:\indian_sign_test")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_class_name(filename):
    # Take ONLY the text before the first dot.
    # Example: Arrest1.xxxx.jpg -> Arrest1
    # We will merge names like Arrest/Arrest1 in a SECOND step.
    name = Path(filename).name
    if "." not in name:
        return Path(name).stem.strip()
    return name.split(".", 1)[0].strip()


def main():
    print("\n" + "=" * 70)
    print("INDIAN SIGN LANGUAGE - INITIAL DATASET SEGREGATION")
    print("=" * 70)

    print("Source      :", SOURCE_DIR)
    print("Destination :", DEST_DIR)

    if not SOURCE_DIR.exists():
        print("\nERROR: Source folder was not found.")
        print("Change SOURCE_DIR at the top of this script if needed.")
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # The screenshot shows the images directly inside this folder.
    image_files = [
        p for p in SOURCE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\nImages found: {len(image_files):,}")

    if not image_files:
        print("No images found directly in SOURCE_DIR.")
        print("If your images are inside another nested folder, tell me.")
        return

    class_files = defaultdict(list)

    for image_path in image_files:
        class_name = get_class_name(image_path.name)
        if class_name:
            class_files[class_name].append(image_path)

    print(f"Classes found: {len(class_files):,}")
    print("\nCopying images...")

    copied = 0
    skipped = 0

    for class_name in sorted(class_files):
        class_dir = DEST_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        files = class_files[class_name]
        class_copied = 0

        for source_file in files:
            destination_file = class_dir / source_file.name

            if destination_file.exists():
                skipped += 1
                continue

            shutil.copy2(source_file, destination_file)
            copied += 1
            class_copied += 1

        print(f"{class_name:<35} {len(files):>6} images")

    print("\n" + "=" * 70)
    print("INITIAL SEGREGATION COMPLETE")
    print("=" * 70)
    print(f"Classes created : {len(class_files):,}")
    print(f"Images found    : {len(image_files):,}")
    print(f"Images copied   : {copied:,}")
    print(f"Images skipped  : {skipped:,}")
    print(f"\nDataset created at:\n{DEST_DIR}")

    print("\nIMPORTANT:")
    print("This is ONLY the first segregation.")
    print("We have NOT merged Arrest/Arrest1/Arrest2 or similar names yet.")
    print("We will do that in the next step after checking the class list.")


if __name__ == "__main__":
    main()
