from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).parent

SOURCE_DIR = ROOT / "source"
TARGET_DIR = ROOT / "target"

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_pixels(directory):
    pixels = []

    for img_file in directory.rglob("*.png"):
        img = np.array(Image.open(img_file), dtype=np.float32) / 255.0
        pixels.append(img.flatten())

    return np.concatenate(pixels)


source_pixels = load_pixels(SOURCE_DIR)
target_pixels = load_pixels(TARGET_DIR)


plt.figure(figsize=(8, 4))

plt.hist(
    source_pixels,
    bins=50,
    alpha=0.6,
    density=True,
    label="Source"
)

plt.hist(
    target_pixels,
    bins=50,
    alpha=0.6,
    density=True,
    label="Target"
)

plt.xlabel("Pixel intensity")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR / "dataset_histograms.png",
    dpi=300
)

plt.close()


means = [
    source_pixels.mean(),
    target_pixels.mean()
]

stds = [
    source_pixels.std(),
    target_pixels.std()
]

plt.figure(figsize=(6, 4))

x = np.arange(2)

plt.bar(x - 0.15, means, width=0.3, label="Mean")
plt.bar(x + 0.15, stds, width=0.3, label="Std")

plt.xticks(x, ["Source", "Target"])

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR / "dataset_statistics.png",
    dpi=300
)

plt.close()

print("Saved:")
print(FIG_DIR / "dataset_histograms.png")
print(FIG_DIR / "dataset_statistics.png")
