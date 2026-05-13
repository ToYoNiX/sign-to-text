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

from features import extract_from_dict, recompute_from_raw63

DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")

# ──────────────────────────────────────────────────────────────────────────────
# Augmentation helpers  (perturb raw 63-coord slice, then recompute full 86)
# ──────────────────────────────────────────────────────────────────────────────

_DORSAL_OCCLUDED = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20]
_FINGER_CHAINS = [
    (0, [1, 2, 3, 4]),
    (5, [6, 7, 8]),
    (9, [10, 11, 12]),
    (13, [14, 15, 16]),
    (17, [18, 19, 20]),
]


def _noise(lm: np.ndarray) -> np.ndarray:
    return lm + np.random.normal(0, 0.012, lm.shape)


def _rotate(lm: np.ndarray) -> np.ndarray:
    ax = np.radians(np.random.uniform(-20, 20))
    ay = np.radians(np.random.uniform(-20, 20))
    az = np.radians(np.random.uniform(-15, 15))
    # X-axis rotation
    cy, sy = np.cos(ax), np.sin(ax)
    y2 = lm[:, 1] * cy - lm[:, 2] * sy
    z2 = lm[:, 1] * sy + lm[:, 2] * cy
    lm = lm.copy()
    lm[:, 1], lm[:, 2] = y2, z2
    # Y-axis rotation
    cy, sy = np.cos(ay), np.sin(ay)
    x2 = lm[:, 0] * cy + lm[:, 2] * sy
    z2 = -lm[:, 0] * sy + lm[:, 2] * cy
    lm[:, 0], lm[:, 2] = x2, z2
    # Z-axis rotation
    cz, sz = np.cos(az), np.sin(az)
    x2 = lm[:, 0] * cz - lm[:, 1] * sz
    y2 = lm[:, 0] * sz + lm[:, 1] * cz
    lm[:, 0], lm[:, 1] = x2, y2
    return lm


def _flip_depth(lm: np.ndarray) -> np.ndarray:
    lm = lm.copy()
    lm[:, 0] *= -1
    lm[:, 2] *= -1
    lm[_DORSAL_OCCLUDED] += np.random.normal(0, 0.04, lm[_DORSAL_OCCLUDED].shape)
    return lm


def _scale(lm: np.ndarray) -> np.ndarray:
    lm = lm * np.random.uniform(0.92, 1.08)
    for root, joints in _FINGER_CHAINS:
        f = np.random.uniform(0.82, 1.18)
        for j in joints:
            lm[j] = lm[root] + (lm[j] - lm[root]) * f
    return lm


def augment_sample(x: np.ndarray, n: int = 15) -> list:
    """
    Generate n augmented 86-float vectors from one sample.
    Augmentation perturbs only the raw 63 coords (first slice of x),
    then recompute_from_raw63 rebuilds all 86 features so orientation
    features are always consistent with the perturbed landmarks.
    """
    lm_orig = x[:63].reshape(21, 3)
    copies = []
    for _ in range(n):
        lm = lm_orig.copy()
        if np.random.rand() > 0.4:
            lm = _noise(lm)
        if np.random.rand() > 0.5:
            lm = _rotate(lm)
        if np.random.rand() > 0.6:
            lm = _flip_depth(lm)
        lm = _scale(lm)
        copies.append(recompute_from_raw63(lm.flatten()))
    return copies


# ──────────────────────────────────────────────────────────────────────────────


def load_dataset(dataset_dir: Path):
    X, y = [], []
    for f in sorted(dataset_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        X.append(extract_from_dict(d))
        y.append(d["label"])
    return np.array(X, dtype=np.float32), y


def plot_confusion(y_true, y_pred, labels, path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), max(7, len(labels) * 0.6)))
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
    print(f"  Confusion matrix → {path}")


def train(augment_copies: int = 15):
    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")
    X_real, y_real = load_dataset(DATASET_DIR)
    print(f"  {len(X_real)} samples, {X_real.shape[1]} features, {len(set(y_real))} classes")

    if augment_copies > 0:
        print(f"Augmenting ({augment_copies}× per sample)...")
        X_aug, y_aug = [], []
        for x, label in zip(X_real, y_real):
            copies = augment_sample(x, n=augment_copies)
            X_aug.extend(copies)
            y_aug.extend([label] * len(copies))
        X_train = np.vstack([X_real, np.array(X_aug, dtype=np.float32)])
        y_train = list(y_real) + y_aug
        print(f"  Training set: {len(X_train)} samples")
    else:
        X_train, y_train = X_real, list(y_real)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    label_map = {str(i): lbl for i, lbl in enumerate(le.classes_)}
    with open(MODELS_DIR / "label_map.json", "w", encoding="utf-8") as fp:
        json.dump(label_map, fp, ensure_ascii=False, indent=2)
    print(f"  Label map saved ({len(label_map)} classes)")

    print("Fitting scaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    X_real_sc = scaler.transform(X_real)
    y_real_enc = le.transform(y_real)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── SVM ───────────────────────────────────────────────────────────────────
    print("\nTraining SVM...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    svm.fit(X_scaled, y_enc)
    joblib.dump(svm, MODELS_DIR / "svm.pkl")
    svm_cv = cross_val_score(
        SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        X_real_sc,
        y_real_enc,
        cv=cv,
        scoring="accuracy",
    )
    print(f"  5-fold CV (real data): {svm_cv.mean():.3f} ± {svm_cv.std():.3f}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y_enc)
    joblib.dump(rf, MODELS_DIR / "rf.pkl")
    rf_cv = cross_val_score(
        RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        X_real_sc,
        y_real_enc,
        cv=cv,
        scoring="accuracy",
    )
    print(f"  5-fold CV (real data): {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")

    # ── Reports & confusion matrices ──────────────────────────────────────────
    classes_str = le.classes_.tolist()
    for name, model in [("SVM", svm), ("RF", rf)]:
        print(f"\n── {name} Classification Report ──")
        y_pred = le.inverse_transform(model.predict(X_scaled))
        print(classification_report(y_train, y_pred, labels=classes_str, zero_division=0))

    plot_confusion(
        y_train,
        le.inverse_transform(svm.predict(X_scaled)),
        classes_str,
        MODELS_DIR / "svm_confusion.png",
    )
    plot_confusion(
        y_train,
        le.inverse_transform(rf.predict(X_scaled)),
        classes_str,
        MODELS_DIR / "rf_confusion.png",
    )
    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--augment", type=int, default=15, metavar="N")
    group.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()
    train(augment_copies=0 if args.no_augment else args.augment)
