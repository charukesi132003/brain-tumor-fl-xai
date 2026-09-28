import kagglehub
from pathlib import Path
import shutil

print("=" * 60)
print("BRAIN TUMOR MRI DATASET DOWNLOADER")
print("=" * 60)

dataset = "masoudnickparvar/brain-tumor-mri-dataset"

print("\nDownloading dataset...")
print("Dataset:", dataset)

path = kagglehub.dataset_download(dataset)

source = Path(path)
destination = Path("data")

print("\nDownloaded to:")
print(source)

print("\nCopying dataset into project/data ...")

destination.mkdir(parents=True, exist_ok=True)

for item in source.iterdir():

    target = destination / item.name

    if item.is_dir():

        shutil.copytree(
            item,
            target,
            dirs_exist_ok=True,
        )

    else:

        shutil.copy2(
            item,
            target,
        )

print("\nDataset ready:")
print(destination.resolve())

print("\nFolders:")

for item in destination.iterdir():

    print(" -", item.name)

print("\nDONE")
