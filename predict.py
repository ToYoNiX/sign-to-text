"""
Real-time ARSL prediction system.

Two modes:

  1. API server (default) — FastAPI, listens on http://localhost:8000
       POST /predict
         Body: { "landmarks": [{x, y, z}, ... 21 points] }
         Response: { "label": "ز", "confidence": 0.97, "model": "svm",
                     "top3": [["ز", 0.97], ["ر", 0.02], ["ذ", 0.01]] }

       GET /health → { "status": "ok", "classes": 38 }

  2. Interactive terminal (--interactive / -i)
       Paste a JSON array of 21 landmarks and press Enter to see predictions.
       Type 'q' to quit.

Usage:
  python predict.py               # start API server on :8000
  python predict.py --port 9000   # custom port
  python predict.py -i            # interactive terminal mode
  python predict.py --model rf    # use Random Forest (default: svm)
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path("models")


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────


def load_models():
    svm = joblib.load(MODELS_DIR / "svm.pkl")
    rf = joblib.load(MODELS_DIR / "rf.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    with open(MODELS_DIR / "label_map.json", encoding="utf-8") as fp:
        label_map = json.load(fp)
    # index → label (label_map already has string keys)
    idx_to_label = {int(k): v for k, v in label_map.items()}
    return svm, rf, scaler, idx_to_label


# ──────────────────────────────────────────────
# Prediction logic
# ──────────────────────────────────────────────


def landmarks_to_features(landmarks: list[dict]) -> np.ndarray:
    """Convert a list of 21 {x, y, z} dicts to a (1, 63) feature array."""
    flat = []
    for lm in landmarks:
        flat.extend([lm["x"], lm["y"], lm["z"]])
    return np.array(flat, dtype=np.float32).reshape(1, -1)


def predict_landmarks(
    landmarks: list[dict],
    model,
    scaler,
    idx_to_label: dict,
    top_k: int = 3,
) -> dict:
    X = landmarks_to_features(landmarks)
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0]
    top_indices = np.argsort(proba)[::-1][:top_k]

    top3 = [[idx_to_label[i], round(float(proba[i]), 4)] for i in top_indices]
    label = top3[0][0]
    confidence = top3[0][1]

    return {"label": label, "confidence": confidence, "top3": top3}


# ──────────────────────────────────────────────
# Interactive terminal mode
# ──────────────────────────────────────────────


def run_interactive(model_name: str = "svm"):
    print(f"Loading models from {MODELS_DIR}/...")
    svm, rf, scaler, idx_to_label = load_models()
    model = svm if model_name == "svm" else rf
    n_classes = len(idx_to_label)
    print(f"Ready. Using {model_name.upper()}, {n_classes} classes.\n")
    print('Paste 21 landmarks as JSON array → [{"x":0.1,"y":0.2,"z":0.0}, ...]')
    print("Type 'q' to quit.\n")

    while True:
        try:
            line = input("landmarks> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if line.lower() in ("q", "quit", "exit"):
            print("Bye.")
            break
        if not line:
            continue

        try:
            landmarks = json.loads(line)
            if not isinstance(landmarks, list) or len(landmarks) != 21:
                got = len(landmarks) if isinstance(landmarks, list) else type(landmarks).__name__
                print(f"  Error: expected a list of 21 points, got {got}")
                continue
            result = predict_landmarks(landmarks, model, scaler, idx_to_label)
            print(f"  → {result['label']}  ({result['confidence'] * 100:.1f}%)")
            top3_str = "  |  ".join(f"{lbl} {conf * 100:.1f}%" for lbl, conf in result["top3"])
            print(f"     Top 3: {top3_str}")
        except json.JSONDecodeError as e:
            print(f"  JSON error: {e}")
        except Exception as e:
            print(f"  Error: {e}")


# ──────────────────────────────────────────────
# FastAPI server mode
# ──────────────────────────────────────────────


def run_server(port: int = 8000, model_name: str = "svm"):
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        print("Install FastAPI and uvicorn to run server mode:")
        print("  pip install fastapi uvicorn")
        sys.exit(1)

    print(f"Loading models from {MODELS_DIR}/...")
    svm, rf, scaler, idx_to_label = load_models()

    app = FastAPI(title="ARSL Prediction API", version="1.0")

    class Landmark(BaseModel):
        x: float
        y: float
        z: float

    class PredictRequest(BaseModel):
        landmarks: list[Landmark]
        model: str = model_name  # "svm" or "rf"

    class PredictResponse(BaseModel):
        label: str
        confidence: float
        model: str
        top3: list[list]

    @app.get("/health")
    def health():
        return {"status": "ok", "classes": len(idx_to_label), "active_model": model_name}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if len(req.landmarks) != 21:
            raise HTTPException(400, f"Expected 21 landmarks, got {len(req.landmarks)}")

        chosen = svm if req.model == "svm" else rf
        lm_dicts = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in req.landmarks]

        result = predict_landmarks(lm_dicts, chosen, scaler, idx_to_label)
        return {**result, "model": req.model}

    print(f"Starting server on http://localhost:{port}")
    print(f"Active model: {model_name.upper()}")
    print(f"Classes: {len(idx_to_label)}")
    print("\nExample request:")
    print(f"  curl -X POST http://localhost:{port}/predict \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"landmarks": [{"x":0,"y":0,"z":0}, ...21 points]}\'\n')

    uvicorn.run(app, host="0.0.0.0", port=port)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARSL real-time prediction")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run interactive terminal mode instead of API server",
    )
    parser.add_argument("--port", type=int, default=8000, help="API server port (default 8000)")
    parser.add_argument(
        "--model", choices=["svm", "rf"], default="svm", help="Which model to use (default: svm)"
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive(model_name=args.model)
    else:
        run_server(port=args.port, model_name=args.model)
