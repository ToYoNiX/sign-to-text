"""
ARSL Live Recognition — web server + prediction API

  python serve.py              # starts on http://localhost:8080
  python serve.py --port 9000

Open http://localhost:8080 in a browser. Allow camera access. Show your hand.

Endpoints:
  GET  /                  browser UI (SVM + RF side-by-side, live camera)
  POST /predict           HTTP — accepts 21 landmarks, returns {svm, rf} results
  WS   /predict?token=…   WebSocket — same format, persistent connection for real-time use

Auth:
  On first run auth.json is generated next to this file and the token is printed.
  The WebSocket endpoint requires ?token=<value> — HTTP POST is open (used by the UI).
  Delete auth.json and restart to rotate the token.
"""

import argparse
import json
import secrets
from pathlib import Path

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from features import extract  # ← single source of truth, 86 features

MODELS_DIR = Path("models")
TEMPLATES_DIR = Path(__file__).parent / "templates"
HTML_PATH = TEMPLATES_DIR / "index.html"
AUTH_FILE = Path(__file__).parent / "auth.json"

# ── Auth token ─────────────────────────────────────────────────────────────────
if not AUTH_FILE.exists():
    _token = secrets.token_urlsafe(32)
    AUTH_FILE.write_text(json.dumps({"token": _token}, indent=2))
    print("=" * 52)
    print(f"  Auth token generated — auth.json")
    print(f"  Token: {_token}")
    print("  Delete auth.json to regenerate.")
    print("=" * 52)
else:
    _token = json.loads(AUTH_FILE.read_text())["token"]

TOKEN = _token

# ── Load models ────────────────────────────────────────────────────────────────
svm = joblib.load(MODELS_DIR / "svm.pkl")
rf = joblib.load(MODELS_DIR / "rf.pkl", mmap_mode="r")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
with open(MODELS_DIR / "label_map.json", encoding="utf-8") as _f:
    IDX_TO_LABEL: dict[int, str] = {int(k): v for k, v in json.load(_f).items()}

app = FastAPI(title="ARSL Live Recognition")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"]
)


# ── Shared prediction helper ───────────────────────────────────────────────────
def _predict(landmarks_raw: list[dict]) -> dict:
    """
    landmarks_raw: list of 21 {x, y, z} dicts (already wrist-normalised by browser)
    Returns dict with 'svm' and 'rf' keys.
    """
    feat = extract(landmarks_raw)  # (86,)
    X = feat.reshape(1, -1)
    X_sc = scaler.transform(X)
    out = {}
    for name, model in [("svm", svm), ("rf", rf)]:
        proba = model.predict_proba(X_sc)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        out[name] = {
            "label": IDX_TO_LABEL[int(top3_idx[0])],
            "confidence": float(proba[top3_idx[0]]),
            "top3": [[IDX_TO_LABEL[int(i)], float(proba[i])] for i in top3_idx],
        }
    return out


# ── HTTP endpoint ──────────────────────────────────────────────────────────────
class Landmark(BaseModel):
    x: float
    y: float
    z: float


class PredictRequest(BaseModel):
    landmarks: list[Landmark]


@app.post("/predict")
def predict_http(req: PredictRequest):
    if len(req.landmarks) != 21:
        return {"error": f"expected 21 landmarks, got {len(req.landmarks)}"}
    lm_dicts = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in req.landmarks]
    return _predict(lm_dicts)


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/predict")
async def predict_ws(ws: WebSocket, token: str = Query(default="")):
    if token != TOKEN:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
                lms = data.get("landmarks", [])
                if len(lms) != 21:
                    await ws.send_text(
                        json.dumps({"error": f"expected 21 landmarks, got {len(lms)}"})
                    )
                    continue
                result = _predict(lms)
                await ws.send_text(json.dumps(result, ensure_ascii=False))
            except (json.JSONDecodeError, KeyError) as e:
                await ws.send_text(json.dumps({"error": str(e)}))
    except WebSocketDisconnect:
        pass


# ── Static assets ──────────────────────────────────────────────────────────────
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
    return Response(content=svg, media_type="image/svg+xml")


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
