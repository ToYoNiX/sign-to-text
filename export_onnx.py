"""
Converts the trained SVM + StandardScaler to a single ONNX file.

Run once after training:
  python export_onnx.py

Outputs (repo root, committed):
  model.onnx      — SVM pipeline (scaler baked in)
  label_map.json  — { "0": "ا", ... } index → Arabic label

After exporting, run `python build.py` to assemble the static site,
or just push — GitHub Actions will deploy automatically.
"""
import json
import shutil
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

from features import extract_from_dict, N_FEATURES   

MODELS_DIR = Path("models")
ROOT       = Path(__file__).parent

# ── Load ───────────────────────────────────────────────────────────────────────
svm    = joblib.load(MODELS_DIR / "svm.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
with open(MODELS_DIR / "label_map.json", encoding="utf-8") as f:
    label_map = json.load(f)

# ── Convert ────────────────────────────────────────────────────────────────────
pipe = Pipeline([("scaler", scaler), ("svm", svm)])
onnx_bytes = convert_sklearn(
    pipe,
    initial_types=[("float_input", FloatTensorType([None, N_FEATURES]))],
    options={id(svm): {"zipmap": False}},
).SerializeToString()

onnx_path = ROOT / "model.onnx"
onnx_path.write_bytes(onnx_bytes)
print(f"model.onnx     → {len(onnx_bytes)/1024:.1f} KB  (input: {N_FEATURES} features)")

label_path = ROOT / "label_map.json"
shutil.copy(MODELS_DIR / "label_map.json", label_path)
print(f"label_map.json → {len(label_map)} classes")

# ── Sanity check: round-trip with a real sample ────────────────────────────────
sess = rt.InferenceSession(onnx_bytes)

# Find any sample in dataset/ to test with
dataset = Path("dataset")
test_files = sorted(dataset.glob("*.json"))
if not test_files:
    print("\nNo dataset files found — skipping sanity check (run build_dataset.py first)")
else:
    test_path = test_files[0]
    with open(test_path, encoding="utf-8") as f:
        sample = json.load(f)
    feat = extract_from_dict(sample).reshape(1, -1)   # (1, 86)

    pred_label, proba = sess.run(None, {"float_input": feat})
    idx      = int(pred_label[0])
    top3_idx = np.argsort(proba[0])[::-1][:3]
    expected = sample["label"]
    predicted = label_map[str(idx)]
    ok = "✓" if predicted == expected else "✗ MISMATCH"
    print(f"\nSanity check on '{test_path.name}':")
    print(f"  Expected:  {expected}")
    print(f"  Predicted: {predicted}  ({proba[0][idx]*100:.1f}%)  {ok}")
    print(f"  Top 3: {[(label_map[str(i)], f'{proba[0][i]*100:.1f}%') for i in top3_idx]}")

print(f"\nDone. ")