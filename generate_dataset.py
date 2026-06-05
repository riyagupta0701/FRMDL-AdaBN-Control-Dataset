import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Reproducibility
SEED = 42
rng  = np.random.default_rng(SEED)

# Dataset parameters
IMG_SIZE    = 32           # 32x32 single-channel images
N_PER_CLASS = 1000          # images per class per domain
CLASSES     = ['circle', 'square', 'triangle']
N_CLASSES   = len(CLASSES)

# Source-domain intensity distribution.
# All classes share identical statistics so that shape remains the only label-related signal.
SOURCE_STATS = {
    'circle':   {'mean': 0.45, 'std': 0.08},
    'square':   {'mean': 0.45, 'std': 0.08},
    'triangle': {'mean': 0.45, 'std': 0.08},
}

# Target-domain intensity distribution.
# The target domain differs only in intensity statistics, creating a controlled BN-relevant domain shift.
TARGET_STATS = {
    'circle':   {'mean': 0.75, 'std': 0.14},
    'square':   {'mean': 0.75, 'std': 0.14},
    'triangle': {'mean': 0.75, 'std': 0.14},
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Class identity is determined solely by geometric shape.
# Domain identity is determined solely by intensity statistics.
def draw_shape(class_name: str, bg_mean: float, bg_std: float,
               size: int = IMG_SIZE) -> np.ndarray:
    """
    Render a geometric shape onto a Gaussian-noise background.

    Background pixels ~ N(bg_mean, bg_std), clipped to [0, 1].
    Foreground (shape) is filled with min(bg_mean + 0.25, 1.0),
    guaranteeing the shape is always visible but the background
    statistics dominate the activation distribution.
    Shape size and centre are randomly jittered slightly so
    the model cannot exploit a fixed spatial prior.

    Parameters
    ----------
    class_name : 'circle' | 'square' | 'triangle'
    bg_mean    : background pixel mean
    bg_std     : background pixel standard deviation
    size       : image width = height in pixels

    Returns
    -------
    float32 ndarray of shape (size, size), values in [0, 1]
    """
    img = rng.normal(bg_mean, bg_std, (size, size)).clip(0.0, 1.0)
    fg  = float(np.clip(bg_mean + 0.25, 0.0, 1.0))

    cx = size // 2 + int(rng.uniform(-size * 0.08, size * 0.08))
    cy = size // 2 + int(rng.uniform(-size * 0.08, size * 0.08))
    r  = int(size * rng.uniform(0.22, 0.32))

    if class_name == 'circle':
        Y, X = np.ogrid[:size, :size]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
        img[mask] = fg

    elif class_name == 'square':
        x0 = max(0, cx - r);  x1 = min(size, cx + r)
        y0 = max(0, cy - r);  y1 = min(size, cy + r)
        img[y0:y1, x0:x1] = fg

    elif class_name == 'triangle':
        apex_y = cy - r
        base_y = cy + r
        for row in range(max(0, apex_y), min(size, base_y + 1)):
            t        = (row - apex_y) / max(base_y - apex_y, 1)
            half_w   = int(r * t)
            col_l    = cx - half_w
            col_r    = cx + half_w
            img[row, max(0, col_l):min(size, col_r)] = fg

    return img.astype(np.float32)


# Domain generation
def generate_domain(domain_name: str, stats: dict, root: str):
    """Write all images for one domain and return path records."""
    records = []
    for label_idx, cls in enumerate(CLASSES):
        out_dir = os.path.join(
            root, domain_name, f'class_{label_idx}_{cls}')
        os.makedirs(out_dir, exist_ok=True)
        m = stats[cls]['mean']
        s = stats[cls]['std']
        for i in range(N_PER_CLASS):
            arr   = draw_shape(cls, m, s)
            u8    = (arr * 255).clip(0, 255).astype(np.uint8)
            fname = f'img_{i:04d}.png'
            fpath = os.path.join(out_dir, fname)
            Image.fromarray(u8, mode='L').save(fpath)
            records.append({
                'path':   os.path.relpath(fpath, root),
                'domain': domain_name,
                'class':  cls,
                'label':  label_idx,
            })
    return records


# Figure 1 -- domain shift visualisation
def make_shift_figure(root: str):
    """Side-by-side grid: source (row 0) vs target (row 1), 3 classes."""
    fig, axes = plt.subplots(2, N_CLASSES, figsize=(9, 6))

    domain_rows = [('Source', SOURCE_STATS), ('Target', TARGET_STATS)]
    for row, (dom_label, stats) in enumerate(domain_rows):
        for col, cls in enumerate(CLASSES):
            ax = axes[row][col]
            sample = draw_shape(cls, stats[cls]['mean'], stats[cls]['std'])
            ax.imshow(sample, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row == 0:
                ax.set_title(cls.capitalize(), fontsize=13, fontweight='bold', pad=6)
            if col == 0:
                ax.set_ylabel(dom_label, fontsize=11, labelpad=8)

    fig.suptitle('Shape identity is unchanged, only pixel statistics shift', fontsize=12, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(root, 'figures', 'domain_shift_vis.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# Figure 2 -- BN statistics comparison
def make_bn_stats_figure(root: str):
    """
    Simulate BN layer statistics for source vs target.
    Draws N images per domain, computes per-image mean and std
    (proxy for what a BN layer accumulates), and plots histograms.
    This directly visualises what AdaBN corrects.
    """
    n_sim = 300
    src_m, src_s, tgt_m, tgt_s = [], [], [], []

    for cls in CLASSES:
        sm = SOURCE_STATS[cls]['mean']; ss = SOURCE_STATS[cls]['std']
        tm = TARGET_STATS[cls]['mean']; ts = TARGET_STATS[cls]['std']
        for _ in range(n_sim // N_CLASSES):
            si = draw_shape(cls, sm, ss).ravel()
            ti = draw_shape(cls, tm, ts).ravel()
            src_m.append(si.mean()); src_s.append(si.std())
            tgt_m.append(ti.mean()); tgt_s.append(ti.std())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    kw_src = dict(color='#4c9be8', alpha=0.75, bins=35, density=True)
    kw_tgt = dict(color='#e8844c', alpha=0.75, bins=35, density=True)

    for ax, sv, tv, xlabel in [
        (ax1, src_m, tgt_m, 'Batch-level mean (mu)'),
        (ax2, src_s, tgt_s, 'Batch-level std (sigma)'),
    ]:
        ax.hist(sv, **kw_src, label='Source domain')
        ax.hist(tv, **kw_tgt, label='Target domain')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.tick_params(colors='#888888')
        ax.set_title(xlabel, fontsize=11, pad=6)

    fig.suptitle(
        'BN Statistics: Source vs Target\n'
        'Mismatched statistics degrade inference; AdaBN corrects this',
        fontsize=12, y=1.03)
    fig.tight_layout()
    out = os.path.join(root, 'figures', 'bn_stats_vis.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    print('Generating source domain ...')
    src_records = generate_domain('source', SOURCE_STATS, BASE_DIR)
    print(f'  {len(src_records)} images written.')

    print('Generating target domain ...')
    tgt_records = generate_domain('target', TARGET_STATS, BASE_DIR)
    print(f'  {len(tgt_records)} images written.')

    print('Generating figures ...')
    os.makedirs(os.path.join(BASE_DIR, 'figures'), exist_ok=True)
    make_shift_figure(BASE_DIR)
    make_bn_stats_figure(BASE_DIR)

    # Store the parameters and measured statistics used to create the synthetic
    # domains so that the generated domain shift is fully reproducible.
    metadata = {
        'description': (
            'Controlled dataset for AdaBN (Li et al., arXiv:1603.04779). '
            'Domain shift is ONLY in pixel intensity statistics (mean, '
            'variance). Shape class identity is identical across domains. '
            'This isolates the property: BN statistics encode domain '
            'identity, and mismatching them causes accuracy loss.'
        ),
        'paper':        'Li et al., arXiv:1603.04779',
        'seed':         SEED,
        'img_size':     IMG_SIZE,
        'n_per_class':  N_PER_CLASS,
        'classes':      CLASSES,
        'source_stats': SOURCE_STATS,
        'target_stats': TARGET_STATS,
        "class_counts": {
            "circle": N_PER_CLASS,
            "square": N_PER_CLASS,
            "triangle": N_PER_CLASS
        },
        'total_images': len(src_records) + len(tgt_records),
        'example_records': src_records[:3] + tgt_records[:3],
    }
    meta_path = os.path.join(BASE_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Metadata written to {meta_path}')
    print('Done.')
