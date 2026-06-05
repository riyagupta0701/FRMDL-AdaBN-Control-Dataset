import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent

SOURCE_DIR = ROOT / "source"
TARGET_DIR = ROOT / "target"


def collect_images(directory):
    imgs = []

    for file in directory.rglob("*.png"):
        arr = np.array(Image.open(file), dtype=np.float32) / 255.0
        imgs.append(arr)

    return np.stack(imgs)


def count_per_class(directory):
    counts = {}

    for class_dir in sorted(directory.iterdir()):
        if class_dir.is_dir():
            counts[class_dir.name] = len(list(class_dir.glob("*.png")))

    return counts


def report(name, directory):
    imgs = collect_images(directory)

    print(f"\n{name}")
    print("-" * 50)

    print("Images:", len(imgs))
    print("Global mean:", imgs.mean())
    print("Global std :", imgs.std())

    print("\nPer-class counts")

    for cls, count in count_per_class(directory).items():
        print(f"  {cls}: {count}")


if __name__ == "__main__":

    report("SOURCE DOMAIN", SOURCE_DIR)
    report("TARGET DOMAIN", TARGET_DIR)

    source = collect_images(SOURCE_DIR)
    target = collect_images(TARGET_DIR)

    print("\nDomain Shift Summary")
    print("-" * 50)

    print("Mean difference:", target.mean() - source.mean())
    print("Std difference :", target.std() - source.std())
    