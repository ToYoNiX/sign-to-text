"""
Builds dataset/ from raw/:
  1. Quality filtering — keeps the best N samples per label (closest to centroid)
     and saves per-label variance heatmaps to dataset/quality/
  2. Mirroring — appends a mirrored copy (x negated) for every mirrorable label

Output layout:
  dataset/
    quality/          ← heatmap PNGs (one per label)
    {label}-1.json    ← best N originals
    ...
    {label}-N+1.json  ← mirrored copies
    ...

Usage:
  python build_dataset.py               # keeps best 30 per label
  python build_dataset.py --keep 50     # keeps best 50
  python build_dataset.py --keep 0      # keep all (no filtering)
"""

import json
import copy
import argparse
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_DIR = Path("raw")
DATASET_DIR = Path("dataset")
QUALITY_DIR = DATASET_DIR / "quality"

LANDMARK_NAMES = [
    "Wrist",
    "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
    "Index MCP", "Index PIP", "Index DIP", "Index Tip",
    "Middle MCP", "Middle PIP", "Middle DIP", "Middle Tip",
    "Ring MCP", "Ring PIP", "Ring DIP", "Ring Tip",
    "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip",
]


def extract_features(d: dict) -> np.ndarray:
    """Flatten all landmarks in all frames to a 1-D feature vector."""
    vecs = []
    for frame in d["frames"]:
        for lm in frame["landmarks"]:
            vecs.extend([lm["x"], lm["y"], lm["z"]])
    return np.array(vecs, dtype=np.float32)


def mirror_sample(d: dict) -> dict:
    m = copy.deepcopy(d)
    for frame in m["frames"]:
        for lm in frame["landmarks"]:
            lm["x"] = -lm["x"]
    m["mirrored"] = True
    return m


def save_heatmap(label: str, features: np.ndarray, selected_mask: np.ndarray, distances: np.ndarray):
    """
    Saves a 2-panel heatmap:
      Left  — per-landmark std-dev (21 × 3) across ALL samples
      Right — per-landmark std-dev across SELECTED samples only
    """
    n_frames = features.shape[1] // 63  # 63 = 21 × 3
    # Use only first frame for static; for dynamic use all frames flattened
    frame_feats = features[:, :63]  # (N, 63) — first frame landmarks

    all_lm = frame_feats.reshape(-1, 21, 3)       # (N, 21, 3)
    sel_lm = frame_feats[selected_mask].reshape(-1, 21, 3)

    all_std = all_lm.std(axis=0)   # (21, 3)
    sel_std = sel_lm.std(axis=0)   # (21, 3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    coord_labels = ["X", "Y", "Z"]

    for ax, data, title in [
        (axes[0], all_std, f"{label} — all {len(features)} samples"),
        (axes[1], sel_std, f"{label} — selected {selected_mask.sum()} samples"),
    ]:
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(3))
        ax.set_xticklabels(coord_labels)
        ax.set_yticks(range(21))
        ax.set_yticklabels(LANDMARK_NAMES, fontsize=8)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label="Std dev")

    plt.suptitle(f"Landmark variance heatmap — {label}", fontsize=12)
    plt.tight_layout()
    safe_name = label.replace("/", "_").replace("\\", "_")
    out_path = QUALITY_DIR / f"{safe_name}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def select_best(features: np.ndarray, keep: int) -> np.ndarray:
    """Return boolean mask for the `keep` samples closest to the centroid."""
    if keep <= 0 or keep >= len(features):
        return np.ones(len(features), dtype=bool)
    centroid = features.mean(axis=0)
    distances = np.linalg.norm(features - centroid, axis=1)
    ranked = np.argsort(distances)
    mask = np.zeros(len(features), dtype=bool)
    mask[ranked[:keep]] = True
    return mask, distances


def build(keep: int = 30):
    DATASET_DIR.mkdir(exist_ok=True)
    QUALITY_DIR.mkdir(exist_ok=True)

    # Group raw files by label
    label_files: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(RAW_DIR.glob("*.json")):
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        label_files[d["label"]].append(d)

    total_written = 0

    for label, samples in sorted(label_files.items()):
        mirrorable = samples[0].get("mirrorable", False)
        features = np.stack([extract_features(d) for d in samples])

        # Quality filtering
        if keep > 0 and keep < len(samples):
            mask, distances = select_best(features, keep)
        else:
            mask = np.ones(len(samples), dtype=bool)
            distances = np.zeros(len(samples))

        selected = [d for d, keep_it in zip(samples, mask) if keep_it]

        # Heatmap
        save_heatmap(label, features, mask, distances)

        idx = 1
        # Originals (selected only)
        for d in selected:
            out = DATASET_DIR / f"{label}-{idx}.json"
            with open(out, "w", encoding="utf-8") as fp:
                json.dump(d, fp, ensure_ascii=False, indent=2)
            idx += 1

        # Mirrored copies
        if mirrorable:
            for d in selected:
                out = DATASET_DIR / f"{label}-{idx}.json"
                with open(out, "w", encoding="utf-8") as fp:
                    json.dump(mirror_sample(d), fp, ensure_ascii=False, indent=2)
                idx += 1

        n_sel = len(selected)
        n_mirror = n_sel if mirrorable else 0
        total_written += n_sel + n_mirror
        print(
            f"  {label:10s}  {len(samples)} raw → {n_sel} selected"
            f" + {n_mirror} mirrored = {n_sel + n_mirror} written"
        )

    print(f"\nDone. {total_written} files written to {DATASET_DIR}/")
    print(f"Quality heatmaps saved to {QUALITY_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep",
        type=int,
        default=30,
        help="Best N samples to keep per label (0 = keep all, default 30)",
    )
    args = parser.parse_args()
    build(keep=args.keep)
