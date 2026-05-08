"""
ARSL Live Recognition — web server + prediction API

  python serve.py              # starts on http://localhost:8080
  python serve.py --port 9000

Open http://localhost:8080 in a browser.
The page opens your camera, runs MediaPipe in the browser, normalises the
landmarks exactly as the capture tool did, and streams them to /predict
which returns both SVM and Random Forest predictions in real time.
"""

import json
import argparse
import numpy as np
import joblib
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

MODELS_DIR    = Path("models")
TEMPLATES_DIR = Path(__file__).parent / "templates"
HTML_PATH     = TEMPLATES_DIR / "index.html"

# ── Load models once at startup ────────────────────────────────────────────────
svm    = joblib.load(MODELS_DIR / "svm.pkl")
rf     = joblib.load(MODELS_DIR / "rf.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
with open(MODELS_DIR / "label_map.json", encoding="utf-8") as _f:
    _raw_map = json.load(_f)
IDX_TO_LABEL: dict[int, str] = {int(k): v for k, v in _raw_map.items()}

app = FastAPI(title="ARSL Live Recognition")


# ── Prediction endpoint ────────────────────────────────────────────────────────
class Landmark(BaseModel):
    x: float
    y: float
    z: float

class PredictRequest(BaseModel):
    landmarks: list[Landmark]

@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.landmarks) != 21:
        return {"error": f"expected 21 landmarks, got {len(req.landmarks)}"}

    flat = []
    for lm in req.landmarks:
        flat.extend([lm.x, lm.y, lm.z])
    X = np.array(flat, dtype=np.float32).reshape(1, -1)
    X_scaled = scaler.transform(X)

    out = {}
    for name, model in [("svm", svm), ("rf", rf)]:
        proba = model.predict_proba(X_scaled)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        out[name] = {
            "label":      IDX_TO_LABEL[int(top3_idx[0])],
            "confidence": float(proba[top3_idx[0]]),
            "top3": [[IDX_TO_LABEL[int(i)], float(proba[i])] for i in top3_idx],
        }
    return out


# ── Static assets ─────────────────────────────────────────────────────────────
@app.get("/style.css", include_in_schema=False)
def stylesheet():
    return FileResponse(TEMPLATES_DIR / "style.css", media_type="text/css")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
        "<rect width='32' height='32' rx='7' fill='%236366f1'/>"
        "<text x='16' y='22' text-anchor='middle' font-size='16' "
        "fill='white' font-family='system-ui'>AR</text></svg>"
    )
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml")

# ── HTML page ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    html = HTML_PATH.read_text(encoding="utf-8")
    return html.replace("%%CONFIG%%", '{"mode":"serve"}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    print(f"Open http://localhost:{args.port} in your browser")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
