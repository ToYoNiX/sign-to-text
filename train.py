"""
Augments, preprocesses, and trains SVM + Random Forest classifiers
on the dataset/ folder produced by build_dataset.py.

Saves to models/:
  svm.pkl         — trained SVM (RBF kernel)
  rf.pkl          — trained Random Forest
  scaler.pkl      — StandardScaler fitted on training data
  label_map.json  — { "0": "ا", "1": "ب", ... }

Usage:
  python train.py
  python train.py --augment 15   # augmented copies per real sample (default 15)
  python train.py --no-augment   # train on raw dataset only
"""

import json
import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")


# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────

def extract_features(d: dict) -> np.ndarray:
    """Flatten landmarks for all frames into a 1-D vector. (Static → 63 floats)"""
    vecs = []
    for frame in d["frames"]:
        for lm in frame["landmarks"]:
            vecs.extend([lm["x"], lm["y"], lm["z"]])
    return np.array(vecs, dtype=np.float32)


# ──────────────────────────────────────────────
# Augmentation helpers
# ──────────────────────────────────────────────

def _reshape(x: np.ndarray) -> np.ndarray:
    """(63,) or (F*63,) → (F, 21, 3)"""
    return x.reshape(-1, 21, 3)


def _flatten(x: np.ndarray) -> np.ndarray:
    """(F, 21, 3) → flat"""
    return x.reshape(-1)


def add_noise(x: np.ndarray, scale: float = 0.012) -> np.ndarray:
    lm = _reshape(x.copy())
    lm += np.random.normal(0, scale, lm.shape)
    return _flatten(lm)


def rotate_3d(x: np.ndarray) -> np.ndarray:
    lm = _reshape(x.copy())
    ax = np.radians(np.random.uniform(-20, 20))   # tilt toward/away camera
    ay = np.radians(np.random.uniform(-20, 20))   # turn left/right
    az = np.radians(np.random.uniform(-15, 15))   # in-plane twist
    # X rotation
    cy, sy = np.cos(ax), np.sin(ax)
    y_new = lm[:, :, 1] * cy - lm[:, :, 2] * sy
    z_new = lm[:, :, 1] * sy + lm[:, :, 2] * cy
    lm[:, :, 1], lm[:, :, 2] = y_new, z_new
    # Y rotation
    cy, sy = np.cos(ay), np.sin(ay)
    x_new =  lm[:, :, 0] * cy + lm[:, :, 2] * sy
    z_new = -lm[:, :, 0] * sy + lm[:, :, 2] * cy
    lm[:, :, 0], lm[:, :, 2] = x_new, z_new
    # Z rotation
    cz, sz = np.cos(az), np.sin(az)
    x_new = lm[:, :, 0] * cz - lm[:, :, 1] * sz
    y_new = lm[:, :, 0] * sz + lm[:, :, 1] * cz
    lm[:, :, 0], lm[:, :, 1] = x_new, y_new
    return _flatten(lm)


# Landmarks occluded when viewing from dorsal side (fingertips + DIP/IP joints)
_DORSAL_OCCLUDED = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20]

def flip_depth(x: np.ndarray, occlusion_noise: float = 0.04) -> np.ndarray:
    """180° Y-axis rotation — simulates dorsal (back-of-hand) view.
    Adds extra noise to landmarks occluded from the dorsal side to match
    MediaPipe's estimation uncertainty for hidden points."""
    lm = _reshape(x.copy())
    lm[:, :, 0] *= -1
    lm[:, :, 2] *= -1
    lm[:, _DORSAL_OCCLUDED, :] += np.random.normal(
        0, occlusion_noise, lm[:, _DORSAL_OCCLUDED, :].shape
    )
    return _flatten(lm)


# Finger chains: (root_landmark_index, [downstream_joint_indices])
_FINGER_CHAINS = [
    (0,  [1, 2, 3, 4]),    # thumb:  wrist → CMC → MCP → IP → Tip
    (5,  [6, 7, 8]),        # index:  MCP → PIP → DIP → Tip
    (9,  [10, 11, 12]),     # middle: MCP → PIP → DIP → Tip
    (13, [14, 15, 16]),     # ring:   MCP → PIP → DIP → Tip
    (17, [18, 19, 20]),     # pinky:  MCP → PIP → DIP → Tip
]

def scale_aug(x: np.ndarray) -> np.ndarray:
    lm = _reshape(x.copy())
    # Mild global scale (hand distance from camera residual)
    lm *= np.random.uniform(0.92, 1.08)
    # Per-finger length variation — simulates proportion differences between people
    for root, joints in _FINGER_CHAINS:
        factor = np.random.uniform(0.82, 1.18)
        for j in joints:
            lm[:, j, :] = lm[:, root, :] + (lm[:, j, :] - lm[:, root, :]) * factor
    return _flatten(lm)


def augment_sample(x: np.ndarray, n: int = 15) -> list[np.ndarray]:
    """Generate `n` augmented copies of feature vector `x`."""
    copies = []
    for _ in range(n):
        aug = x.copy()
        if np.random.rand() > 0.4:
            aug = add_noise(aug)
        if np.random.rand() > 0.5:
            aug = rotate_3d(aug)
        if np.random.rand() > 0.6:
            aug = flip_depth(aug)
        aug = scale_aug(aug)  # always
        copies.append(aug)
    return copies


# ──────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────

def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, list[str]]:
    X, y = [], []
    for f in sorted(dataset_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        X.append(extract_features(d))
        y.append(d["label"])
    return np.array(X, dtype=np.float32), y


# ──────────────────────────────────────────────
# Confusion matrix plot
# ──────────────────────────────────────────────

def plot_confusion(y_true, y_pred, labels: list[str], path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), max(7, len(labels) * 0.55)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.3,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(path.stem.upper() + " — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def train(augment_copies: int = 15):
    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")
    X_real, y_real = load_dataset(DATASET_DIR)
    print(f"  {len(X_real)} samples, {X_real.shape[1]} features, {len(set(y_real))} classes")

    # Augmentation
    if augment_copies > 0:
        print(f"Augmenting ({augment_copies}× per sample)...")
        X_aug, y_aug = [], []
        for x, label in zip(X_real, y_real):
            copies = augment_sample(x, n=augment_copies)
            X_aug.extend(copies)
            y_aug.extend([label] * len(copies))
        X_train = np.vstack([X_real, np.array(X_aug, dtype=np.float32)])
        y_train = y_real + y_aug
        print(f"  Training set after augmentation: {len(X_train)} samples")
    else:
        X_train, y_train = X_real, y_real

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    label_map = {str(i): label for i, label in enumerate(le.classes_)}

    # Save label map
    with open(MODELS_DIR / "label_map.json", "w", encoding="utf-8") as fp:
        json.dump(label_map, fp, ensure_ascii=False, indent=2)
    print(f"  Label map saved ({len(label_map)} classes)")

    # Scale — important for SVM
    print("Fitting scaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # ── Cross-validation on real data (unaugmented, unscaled raw) ──
    X_real_scaled = scaler.transform(X_real)
    y_real_enc = le.transform(y_real)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── SVM ──────────────────────────────────────────────────────────
    print("\nTraining SVM (RBF kernel)...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    svm.fit(X_scaled, y_enc)
    joblib.dump(svm, MODELS_DIR / "svm.pkl")

    svm_cv = cross_val_score(
        SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        X_real_scaled, y_real_enc, cv=cv, scoring="accuracy"
    )
    print(f"  5-fold CV accuracy (real data): {svm_cv.mean():.3f} ± {svm_cv.std():.3f}")
    print(f"  Saved → {MODELS_DIR}/svm.pkl")

    # ── Random Forest ─────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y_enc)
    joblib.dump(rf, MODELS_DIR / "rf.pkl")

    rf_cv = cross_val_score(
        RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        X_real_scaled, y_real_enc, cv=cv, scoring="accuracy"
    )
    print(f"  5-fold CV accuracy (real data): {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")
    print(f"  Saved → {MODELS_DIR}/rf.pkl")

    # ── Classification report on full training set ─────────────────────
    classes_str = le.classes_.tolist()
    print("\n── SVM Classification Report (training set) ──")
    y_pred_svm = le.inverse_transform(svm.predict(X_scaled))
    print(classification_report(y_train, y_pred_svm, labels=classes_str, zero_division=0))

    print("── RF Classification Report (training set) ──")
    y_pred_rf = le.inverse_transform(rf.predict(X_scaled))
    print(classification_report(y_train, y_pred_rf, labels=classes_str, zero_division=0))

    # ── Confusion matrices ─────────────────────────────────────────────
    plot_confusion(y_train, y_pred_svm, classes_str, MODELS_DIR / "svm_confusion.png")
    plot_confusion(y_train, y_pred_rf, classes_str, MODELS_DIR / "rf_confusion.png")

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--augment", type=int, default=15, metavar="N",
                       help="Augmented copies per sample (default 15)")
    group.add_argument("--no-augment", action="store_true",
                       help="Skip augmentation")
    args = parser.parse_args()

    copies = 0 if args.no_augment else args.augment
    train(augment_copies=copies)
