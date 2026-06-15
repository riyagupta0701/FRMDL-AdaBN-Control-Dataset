# AdaBN Control Dataset

Control dataset for testing the role of Batch Normalisation statistics under domain shift. The dataset consists of two domains — *source* and *target* — each containing 3000 labelled greyscale images across three geometric classes.

Based on: <br>
Li et al., *Revisiting Batch Normalisation for Practical Domain Adaptation* — [arXiv:1603.04779](https://arxiv.org/abs/1603.04779) (AdaBN) <br>
Wang et al., *Tent: Fully Test-Time Adaptation by Entropy Minimisation* — [arXiv:2006.10726](https://arxiv.org/abs/2006.10726) (Tent)


## Table of Contents

1. [Overview](#overview)
2. [Motivation & Tested Property](#motivation--tested-property)
3. [Dataset Description](#dataset-description)
4. [Controlled Factors](#controlled-factors)
5. [Repository Structure](#repository-structure)
6. [Quick Start](#quick-start)
7. [Generation Method](#generation-method)
8. [Experiments](#experiments)
9. [Limitations](#limitations)


## Overview

This repository provides a minimal, fully controlled image dataset for empirically testing a central hypothesis motivating Adaptive Batch Normalisation (AdaBN):

Label-related knowledge is primarily stored in the network weights, whereas domain-related information is reflected in Batch Normalisation (BN) statistics, namely the running mean (μ) and variance (σ²) accumulated from training data.

The dataset is designed so that exactly one confounder varies between the source and target domains: the pixel intensity statistics (mean and standard deviation). Shape geometry, class labels, image resolution, and spatial layout are all held constant. This allows a clean, controlled test of whether mismatched BN statistics contribute to accuracy degradation under domain shift and whether correcting them via AdaBN improves performance.

The dataset was created to evaluate the AdaBN adaptation mechanism and to provide a controlled benchmark for comparison with test-time adaptation methods such as Tent.


## Motivation & Tested Property

### Background: Batch Normalisation and domain shift

Batch Normalisation stabilises deep network training by normalising activations to have approximately zero mean and unit variance. It does so by storing BN running statistics (mean μ and variance σ²) computed from the training data, and using those statistics at test time to normalise activations.

When training data (source domain) and test data (target domain) come from different distributions (different cameras, sensors, or lighting conditions) those stored statistics become mismatched to the incoming target activations at every layer of the network. AdaBN hypothesises that mismatched BN statistics contribute significantly to performance degradation under domain shift.

### The property this dataset tests

| | |
|---|---|
| Property | Domain-specific information is reflected in BN statistics (μ, σ²). Mismatched source statistics contribute to domain shift, while re-estimating target statistics can reduce this mismatch. |
| Single confounder | Pixel intensity mean and standard deviation only. All other factors are identical across domains. |
| Control | Source-trained model on source test images → high accuracy. |
| Problem | Source-trained model on target images (source BN stats) → degraded accuracy. |
| Fix | Re-estimate BN stats from unlabelled target images (AdaBN) → accuracy improves. |

### Connection to AdaBN

The motivation comes directly from the pilot experiment in AdaBN (Section 3.1, Figure 2), where BN statistics collected from different domains form clearly separable clusters. This observation suggests that BN statistics capture domain-specific characteristics. AdaBN therefore adapts a network by replacing source-domain BN statistics with statistics computed from the target domain while keeping the learned weights fixed.

### Why this matters for Tent

Tent (Wang et al., 2021) extends AdaBN by additionally optimising the BN affine parameters (γ, β) via entropy minimisation, on top of re-estimating (μ, σ²). Understanding why re-estimating the statistics alone already helps, and by how much, is essential groundwork for understanding the motivation and incremental contribution of Tent. The controlled setting here isolates that step precisely.

### Goal of the Control Dataset

The goal of this control dataset is to isolate the specific mechanism targeted by AdaBN: the mismatch between source-domain and target-domain Batch Normalisation statistics.

Within the control-dataset framework, the dataset is designed to evaluate whether:

- a controlled intensity-based domain shift is a reasonable approximation of a real distribution shift (c1),
- a standard BN-equipped CNN provides a reasonable baseline (c2),
- mismatched BN statistics produce measurable performance degradation (c3), and
- re-estimating BN statistics via AdaBN alleviates this degradation (c4).

By keeping classes, labels, geometry, and class frequencies fixed while varying only image intensity statistics, the dataset provides a direct test of AdaBN's central adaptation mechanism.


## Dataset Description

| Property | Value |
|---|---|
| Total images | 6000 |
| Image size | 32 × 32 pixels, single channel (greyscale) |
| Classes | Circle, Square, Triangle |
| Domains | Source · Target |
| Images per class per domain | 1000 |
| Random seed | 42 (`numpy.random.default_rng(42)`) |
| Format | 8-bit greyscale PNG |

### Dataset Design

The dataset is intentionally designed to isolate a single form of domain shift.

The source and target domains contain:

- identical classes
- identical labels
- identical geometric shapes
- identical image resolution
- identical class frequencies

The only systematic difference between domains is the image intensity distribution.

This design directly targets the AdaBN hypothesis that domain-specific information is reflected in Batch Normalisation statistics.

### Pixel Intensity Parameters

| Domain | Class | μ_bg | σ_bg |
|---|---|---|---|
| Source | Circle | 0.45 | 0.08 |
| Source | Square | 0.45 | 0.08 |
| Source | Triangle | 0.45 | 0.08 |
| Target | Circle | 0.75 | 0.14 |
| Target | Square | 0.75 | 0.14 |
| Target | Triangle | 0.75 | 0.14 |

All classes share identical statistics within a domain. This prevents intensity from leaking class information and ensures that shape remains the only label-related signal.

### Dataset Statistics

The generated dataset was validated using `validate_dataset.py`, which computes empirical means, standard deviations, and class counts directly from the generated images.

#### Global Statistics

| Statistic | Source | Target |
|---|---:|---:|
| Images | 3000 | 3000 |
| Mean intensity | 0.4980 | 0.7968 |
| Standard deviation | 0.1229 | 0.1582 |

#### Measured Domain Shift

| Metric | Difference |
|---|---:|
| Mean shift | +0.2988 |
| Standard deviation shift | +0.0353 |

#### Class Distribution

| Class | Source | Target |
|---|---:|---:|
| Circle | 1000 | 1000 |
| Square | 1000 | 1000 |
| Triangle | 1000 | 1000 |

The class distribution is perfectly balanced across domains.

### Domain shift visualisation

*Source (top row) versus target (bottom row). Shape identity is unchanged while image intensity statistics shift substantially.*

![Domain shift visualisation](figures/domain_shift_vis.png)

## Controlled Factors

A control dataset isolates one specific property while holding all other factors constant. In this dataset:

| Factor | Controlled? |
|----------|----------|
| Class labels | Yes |
| Shape geometry | Yes |
| Class frequencies | Yes |
| Resolution | Yes |
| Shape-to-background contrast | Yes |
| Mean intensity (μ_bg) | No; varies by domain |
| Standard deviation (σ_bg) | No; varies by domain |

The source and target domains differ only in image intensity statistics. Any performance degradation therefore cannot be attributed to changes in semantics, class identity, geometry, or dataset composition. This allows a direct test of the AdaBN hypothesis that mismatched BN statistics contribute to domain shift.


## Repository Structure

```text
FRMDL-AdaBN-Control-Dataset/
│
├── README.md
├── generate_dataset.py
├── validate_dataset.py
├── generate_validation_plots.py
├── experiments.ipynb
├── metadata.json
│
├── source/
│   ├── class_0_circle/
│   ├── class_1_square/
│   └── class_2_triangle/
│
├── target/
│   ├── class_0_circle/
│   ├── class_1_square/
│   └── class_2_triangle/
│
└── figures/
    ├── domain_shift_vis.png
    ├── bn_stats_vis.png
    ├── claim1_domain_shift.png
    ├── claim2_source_accuracy.png
    ├── claims3_4_adabn.png
    ├── dataset_histograms.png
    └── dataset_statistics.png
```


## Quick Start

### Generate the dataset from scratch

```bash
# 1. Clone the repository
git clone https://github.com/riyagupta0701/FRMDL-AdaBN-Control-Dataset.git

# 2. Install dependencies (no GPU required)
pip install numpy pillow matplotlib

# 3. Generate all images and figures
python generate_dataset.py
```

This writes 6000 PNG images to `source/` and `target/`, generates `domain_shift_vis.png` and `bn_stats_vis.png` to `figures/`, and writes `metadata.json`.

```bash
# 4. Run the experiments notebook
pip install jupyter torch torchvision scipy
jupyter notebook experiments.ipynb
```

Run all cells in order. This trains the CNN, applies AdaBN, and saves the experiment figures (`claim1_domain_shift.png`, `claim2_source_accuracy.png`, `claims3_4_adabn.png`) to `figures/`.

```bash
# 5. (Optional) Validate the generated dataset statistics
python validate_dataset.py

# 6. (Optional) Generate additional histogram and statistics plots
python generate_validation_plots.py
```

### Load with PyTorch (example)

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),          # → [0, 1] float32, shape (1, 32, 32)
])

source_dataset = ImageFolder('source/', transform=transform)
target_dataset = ImageFolder('target/', transform=transform)

# Class mapping: {0: 'class_0_circle', 1: 'class_1_square', 2: 'class_2_triangle'}
print(source_dataset.class_to_idx)
```

### Load with NumPy (no framework required)

```python
import numpy as np
from PIL import Image
from pathlib import Path

def load_domain(domain: str):
    """Returns (images, labels) arrays for a domain."""
    root   = Path(domain)
    images, labels = [], []
    for label_idx, cls_dir in enumerate(sorted(root.iterdir())):
        for img_path in sorted(cls_dir.glob('*.png')):
            img = np.array(Image.open(img_path)) / 255.0   # float32 in [0, 1]
            images.append(img)
            labels.append(label_idx)
    return np.stack(images), np.array(labels)

X_src, y_src = load_domain('source')   # shape (3000, 32, 32)
X_tgt, y_tgt = load_domain('target')   # shape (3000, 32, 32)
print(X_src.shape, y_src.shape)
```


## Generation Method

Each image is produced by the following three-step procedure:

**Step 1: Background**  
A 32×32 float array is sampled pixel-wise from a Gaussian distribution x ~ N(μ_bg, σ_bg²) where μ_bg denotes the mean intensity and σ_bg denotes the standard deviation. The parameters (μ_bg, σ_bg) are the only quantities that differ between source and target domains.

**Step 2: Shape**  
A foreground intensity is set to min(μ_bg + 0.25, 1). The geometric shape (determined by the class label) is rasterised at a randomised scale (radius ≈ 22–32% of image width) with a small random centre jitter (±8% of image width) to prevent the model exploiting a fixed spatial prior.

**Step 3: Quantise and save**  
The float array is scaled to [0, 255], cast to uint8, and saved as a greyscale PNG.

**Design rationale:** The +0.25 foreground offset is applied in both domains, so the relative contrast of shape versus background is preserved across domains. The shape is always visible, but the absolute intensity of both foreground and background shifts together with μ_bg. The only systematic difference between domains is the image intensity distribution. Both background and foreground intensities shift together, while shape geometry and class identity remain unchanged. Since all classes share identical statistics within a domain, shape remains the only label-related signal while intensity statistics become the only domain-related signal.


## Experiments

`experiments.ipynb` trains a CNN with BN on the source domain and evaluates it under three conditions to support the four claims.

### Model and training

| Setting | Value |
|---|---|
| Architecture | SimpleCNN: 3× (Conv2d → BN → ReLU → MaxPool2d), then 2× Linear |
| Parameters | ~150K |
| Input | 32×32 greyscale, single channel |
| Training data | Source domain, 80/20 train/test split |
| Epochs | 25 |
| Optimiser | Adam, lr=1e-3 |
| Loss | Cross-entropy |
| Batch size | 64 |

### Evaluation conditions

| Condition | Domain | BN statistics used | Claim tested |
|---|---|---|---|
| Dataset statistics | Source & Target | N/A — pixel-level intensity statistics | 1 |
| Baseline | Source | Source (accumulated during training) | 2 |
| Domain shift | Target | Source (accumulated during training) | 3 |
| AdaBN | Target | Target (re-estimated from unlabelled target data) | 4 |

### AdaBN procedure

AdaBN is applied without modifying the trained weights:

1. Deep-copy the source-trained model.
2. Reset all BN running statistics (`running_mean = 0`, `running_var = 1`).
3. Set `momentum = None` (cumulative moving average over all target batches).
4. Forward-pass all unlabelled target samples in `train()` mode with `torch.no_grad()`.
5. Evaluate in `eval()` mode.

### Output figures

The notebook saves three figures to `figures/`:

| Figure | Contents | Claim |
|---|---|---|
| `claim1_domain_shift.png` | Pixel intensity histograms and Cohen's d | 1 |
| `claim2_source_accuracy.png` | Training curve and source-domain test accuracy | 2 |
| `claims3_4_adabn.png` | Accuracy under all three conditions; shift in BN running means after AdaBN | 3, 4 |


## Limitations

- **Clipping boundary.** The foreground intensity formula `f = min(μ_bg + 0.25, 1)` clips to `f = 1.0` in the target domain (versus `f = 0.70` in the source), introducing a small uncontrolled variation. All three classes are affected equally, so the label-discriminative signal remains intact.
- **Synthetic design.** The controlled shift may not generalise to real-world domain shifts, which typically involve more complex and simultaneous covariate changes (texture, viewpoint, sensor characteristics, object appearance).
- **Task simplicity.** The task is deliberately simple; more demanding tasks with fine-grained categories may exhibit different sensitivity to BN statistics mismatch.
- **Shift magnitude.** The shift is large by design (~3.7σ relative to the source standard deviation). Whether AdaBN remains effective under subtler shifts is not addressed here.
