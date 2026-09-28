"""
Brain Tumor MRI Dataset Verification
"""

from pathlib import Path
from collections import Counter

from PIL import Image


DATA_ROOT = Path("data")

TRAIN_DIR = DATA_ROOT / "Training"
TEST_DIR = DATA_ROOT / "Testing"

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def count_images(directory: Path):

    counts = {}

    for class_directory in sorted(directory.iterdir()):

        if not class_directory.is_dir():
            continue

        files = [
            file
            for file in class_directory.rglob("*")
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        ]

        counts[class_directory.name] = len(files)

    return counts


def check_images(directory: Path):

    corrupt = []
    sizes = Counter()

    for file in directory.rglob("*"):

        if (
            not file.is_file()
            or file.suffix.lower() not in VALID_EXTENSIONS
        ):
            continue

        try:

            with Image.open(file) as image:

                image.verify()

            with Image.open(file) as image:

                sizes[image.size] += 1

        except Exception as exc:

            corrupt.append(
                (
                    str(file),
                    str(exc),
                )
            )

    return corrupt, sizes


def print_counts(title, counts):

    print()
    print(title)
    print("-" * 50)

    total = 0

    for class_name, count in counts.items():

        print(
            f"{class_name:<20}: {count}"
        )

        total += count

    print("-" * 50)

    print(
        f"{'TOTAL':<20}: {total}"
    )

    return total


def main():

    print("=" * 60)
    print("BRAIN TUMOR DATASET VERIFICATION")
    print("=" * 60)

    if not TRAIN_DIR.exists():

        raise FileNotFoundError(
            f"Missing directory: {TRAIN_DIR}"
        )

    if not TEST_DIR.exists():

        raise FileNotFoundError(
            f"Missing directory: {TEST_DIR}"
        )

    train_counts = count_images(
        TRAIN_DIR
    )

    test_counts = count_images(
        TEST_DIR
    )

    train_total = print_counts(
        "TRAINING DATA",
        train_counts,
    )

    test_total = print_counts(
        "TESTING DATA",
        test_counts,
    )

    print()
    print("DATASET SUMMARY")
    print("-" * 50)

    print(
        "Training:",
        train_total,
    )

    print(
        "Testing :",
        test_total,
    )

    print(
        "Total   :",
        train_total + test_total,
    )

    print()
    print("Checking image integrity...")

    train_corrupt, train_sizes = check_images(
        TRAIN_DIR
    )

    test_corrupt, test_sizes = check_images(
        TEST_DIR
    )

    corrupt = (
        train_corrupt
        + test_corrupt
    )

    print()
    print(
        "Corrupted images:",
        len(corrupt),
    )

    if corrupt:

        for filename, error in corrupt[:20]:

            print(
                "CORRUPT:",
                filename,
                error,
            )

    else:

        print(
            "All images opened successfully."
        )

    print()
    print("Most common image sizes:")

    combined_sizes = (
        train_sizes
        + test_sizes
    )

    for size, count in combined_sizes.most_common(10):

        print(
            f"{size}: {count}"
        )

    print()
    print("=" * 60)

    if len(corrupt) == 0:

        print(
            "DATASET VERIFICATION: PASSED"
        )

    else:

        print(
            "DATASET VERIFICATION: WARNING"
        )

    print("=" * 60)


if __name__ == "__main__":

    main()
