"""
Visualize class correlations for trained SVM and Random Forest models.

Loads the real (unaugmented) dataset, runs both models, and produces:
  models/svm_correlation.png   — row-normalised confusion matrix (recall %)
  models/rf_correlation.png    — row-normalised confusion matrix (recall %)
  models/comparison.png        — side-by-side correlation matrices

Also prints the top-N most confused class pairs for each model.

Usage:
  python analyze.py
  python analyze.py --top 15        # show top-15 confused pairs (default 10)
  python analyze.py --dataset-dir dataset
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

MODELS_DIR = Path("models")
DATASET_DIR = Path("dataset")


def extract_features(d: dict) -> np.ndarray:
    vecs = []
    for frame in d["frames"]:
        for lm in frame["landmarks"]:
            vecs.extend([lm["x"], lm["y"], lm["z"]])
    return np.array(vecs, dtype=np.float32)


def load_dataset(dataset_dir: Path):
    X, y = [], []
    for f in sorted(dataset_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        X.append(extract_features(d))
        y.append(d["label"])
    return np.array(X, dtype=np.float32), y


def norm_cm(y_true, y_pred, labels):
    """Row-normalised confusion matrix (each row sums to 1)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


def plot_correlation(cm_norm, labels, title, path):
    n = len(labels)
    size = max(10, n * 0.6)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".0%",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.2,
        cbar_kws={"label": "Recall rate"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_comparison(cm_svm, cm_rf, labels, path):
    n = len(labels)
    w = max(18, n * 0.6)
    fig, axes = plt.subplots(1, 2, figsize=(w * 2, w * 0.85))
    for ax, cm, title in zip(axes, [cm_svm, cm_rf], ["SVM", "Random Forest"]):
        sns.heatmap(
            cm,
            annot=True,
            fmt=".0%",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.2,
            cbar=False,
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(f"{title} — Class Correlation", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def top_confusions(cm_norm, labels, n=10, model_name=""):
    """Print the n most common off-diagonal confusions."""
    cm = cm_norm.copy()
    np.fill_diagonal(cm, 0)
    flat = [
        (cm[i, j], labels[i], labels[j])
        for i in range(len(labels))
        for j in range(len(labels))
        if cm[i, j] > 0
    ]
    flat.sort(reverse=True)
    print(f"\n  Top-{n} confusions for {model_name}:")
    if not flat:
        print("    (no off-diagonal confusions on this dataset)")
        return
    for rate, true_cls, pred_cls in flat[:n]:
        print(f"    {true_cls!r:20s} → predicted as {pred_cls!r:20s}  ({rate:.1%})")


def main(dataset_dir: Path, top_n: int):
    print("Loading models...")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    svm = joblib.load(MODELS_DIR / "svm.pkl")
    rf = joblib.load(MODELS_DIR / "rf.pkl")
    with open(MODELS_DIR / "label_map.json", encoding="utf-8") as fp:
        label_map = json.load(fp)

    labels = [label_map[str(i)] for i in range(len(label_map))]

    print(f"Loading dataset from {dataset_dir}...")
    X, y_true = load_dataset(dataset_dir)
    print(f"  {len(X)} samples, {len(set(y_true))} classes")

    X_scaled = scaler.transform(X)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(labels)
    y_enc = le.transform(y_true)

    print("\nRunning predictions...")
    y_pred_svm = le.inverse_transform(svm.predict(X_scaled))
    y_pred_rf = le.inverse_transform(rf.predict(X_scaled))

    cm_svm = norm_cm(y_true, y_pred_svm, labels)
    cm_rf = norm_cm(y_true, y_pred_rf, labels)

    print("\nGenerating correlation plots...")
    plot_correlation(cm_svm, labels, "SVM — Class Correlation (Recall %)", MODELS_DIR / "svm_correlation.png")
    plot_correlation(cm_rf, labels, "Random Forest — Class Correlation (Recall %)", MODELS_DIR / "rf_correlation.png")
    plot_comparison(cm_svm, cm_rf, labels, MODELS_DIR / "comparison.png")

    top_confusions(cm_svm, labels, n=top_n, model_name="SVM")
    top_confusions(cm_rf, labels, n=top_n, model_name="Random Forest")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10, metavar="N", help="Top-N confused pairs to print (default 10)")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()
    main(args.dataset_dir, args.top)
