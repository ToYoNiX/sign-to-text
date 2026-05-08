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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

MODELS_DIR = Path("models")

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


# ── HTML page ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


# ── HTML / CSS / JS (embedded) ─────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ARSL Live Recognition</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 24px 16px 40px;
  gap: 16px;
}

/* ── header ── */
.top-bar {
  width: 100%; max-width: 940px;
  display: flex; align-items: center; justify-content: space-between; gap: 12;
}
h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }
.badge {
  background: #1e293b; border: 1px solid #334155; border-radius: 9999px;
  padding: 4px 14px; font-size: 12px; color: #94a3b8;
}

/* ── status pill ── */
.status-pill {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 18px; border-radius: 9999px;
  font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  transition: background 0.3s;
}
.status-pill .dot {
  width: 8px; height: 8px; border-radius: 50%; background: rgba(255,255,255,0.7);
}

/* ── camera ── */
.video-wrap {
  position: relative;
  width: 100%; max-width: 940px;
  aspect-ratio: 16/9;
  background: #1e293b;
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 4px 40px rgba(0,0,0,0.55);
}
#video {
  width: 100%; height: 100%;
  object-fit: cover;
  display: block;
  transform: scaleX(-1);
}
#canvas {
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  transform: scaleX(-1);
}
.fps-badge {
  position: absolute; bottom: 12px; right: 14px;
  background: rgba(15,23,42,0.78); backdrop-filter: blur(4px);
  border: 1px solid #334155; border-radius: 9999px;
  padding: 3px 10px; font-size: 11px; font-weight: 700; color: #64748b;
  pointer-events: none;
}

/* ── prediction panels ── */
.predictions {
  width: 100%; max-width: 940px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

.model-card {
  background: #1e293b;
  border-radius: 14px;
  border: 1px solid #334155;
  overflow: hidden;
  transition: border-color 0.3s;
}
.model-card.agreement { border-color: #22c55e; }

.card-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 16px;
  background: #0f172a;
  border-bottom: 1px solid #334155;
  font-size: 13px; font-weight: 700; letter-spacing: 0.3px;
}
.model-name-svm  { color: #a5b4fc; }
.model-name-rf   { color: #86efac; }

.card-conf-badge {
  padding: 2px 10px; border-radius: 9999px;
  font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
  transition: background 0.2s, color 0.2s;
}

.card-body { padding: 16px 20px; display: flex; flex-direction: column; gap: 14px; }

/* big prediction sign */
.big-sign {
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  padding: 14px 0 8px;
}
.sign-arabic {
  font-size: 72px; font-weight: 800; line-height: 1;
  direction: rtl;
  transition: opacity 0.15s;
  min-height: 80px;
  display: flex; align-items: center; justify-content: center;
}
.sign-latin { font-size: 13px; color: #64748b; font-weight: 500; }

/* confidence bar for big sign */
.conf-bar-wrap {
  width: 100%;
  height: 6px; background: #0f172a; border-radius: 9999px; overflow: hidden;
}
.conf-bar-fill {
  height: 100%; border-radius: 9999px;
  transition: width 0.2s ease, background 0.3s;
}

/* top-3 list */
.top3 { display: flex; flex-direction: column; gap: 8px; }
.top3-row {
  display: flex; align-items: center; gap: 10px;
}
.top3-sign {
  font-size: 18px; font-weight: 700; direction: rtl;
  min-width: 28px; text-align: center;
}
.top3-bar-wrap {
  flex: 1; height: 6px; background: #0f172a; border-radius: 9999px; overflow: hidden;
}
.top3-bar-fill {
  height: 100%; border-radius: 9999px; transition: width 0.2s ease;
}
.top3-pct { font-size: 12px; font-weight: 600; min-width: 44px; text-align: right; color: #94a3b8; }

/* consensus row */
.consensus {
  width: 100%; max-width: 940px;
  background: #1e293b;
  border-radius: 12px;
  border: 1px solid #334155;
  padding: 14px 20px;
  display: flex; align-items: center; gap: 14px;
  font-size: 14px;
  transition: border-color 0.3s, background 0.3s;
}
.consensus.agree {
  border-color: #22c55e;
  background: #052e16;
}
.consensus.disagree {
  border-color: #f59e0b;
  background: #1c1400;
}
.consensus-label { color: #64748b; font-weight: 600; flex-shrink: 0; }
.consensus-sign {
  font-size: 28px; font-weight: 800; direction: rtl;
  margin-right: 2px;
}
.consensus-text { color: #94a3b8; font-size: 13px; }

/* error box */
.error-box {
  width: 100%; max-width: 940px;
  background: #7f1d1d; color: #fca5a5;
  border-radius: 10px; padding: 12px 20px;
  font-size: 14px; text-align: center;
}

/* idle state */
.idle { color: #475569; }

@media (max-width: 600px) {
  .predictions { grid-template-columns: 1fr; }
  .sign-arabic  { font-size: 52px; }
}
</style>
</head>
<body>

<div class="top-bar">
  <h1>ARSL Live Recognition</h1>
  <div class="badge" id="model-badge">SVM + Random Forest</div>
</div>

<div class="status-pill" id="status-pill" style="background:#f59e0b">
  <span class="dot"></span>
  <span id="status-text">Loading MediaPipe…</span>
</div>

<div id="error-box" class="error-box" style="display:none"></div>

<div class="video-wrap">
  <video id="video" playsinline muted></video>
  <canvas id="canvas"></canvas>
  <div class="fps-badge" id="fps-badge">— fps</div>
</div>

<div class="predictions">
  <div class="model-card" id="card-svm">
    <div class="card-header">
      <span class="model-name-svm">SVM</span>
      <span class="card-conf-badge" id="svm-conf-badge" style="background:#1e293b;color:#475569">—</span>
    </div>
    <div class="card-body">
      <div class="big-sign">
        <div class="sign-arabic idle" id="svm-sign">—</div>
      </div>
      <div class="conf-bar-wrap">
        <div class="conf-bar-fill" id="svm-bar" style="width:0%;background:#6366f1"></div>
      </div>
      <div class="top3" id="svm-top3"></div>
    </div>
  </div>

  <div class="model-card" id="card-rf">
    <div class="card-header">
      <span class="model-name-rf">Random Forest</span>
      <span class="card-conf-badge" id="rf-conf-badge" style="background:#1e293b;color:#475569">—</span>
    </div>
    <div class="card-body">
      <div class="big-sign">
        <div class="sign-arabic idle" id="rf-sign">—</div>
      </div>
      <div class="conf-bar-wrap">
        <div class="conf-bar-fill" id="rf-bar" style="width:0%;background:#22c55e"></div>
      </div>
      <div class="top3" id="rf-top3"></div>
    </div>
  </div>
</div>

<div class="consensus" id="consensus">
  <span class="consensus-label">Consensus</span>
  <span class="consensus-sign idle" id="consensus-sign">—</span>
  <span class="consensus-text" id="consensus-text">waiting for hand…</span>
</div>

<script type="module">
import { HandLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/+esm";

// ── MediaPipe hand skeleton topology ──────────────────────────────────────────
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

// ── DOM refs ───────────────────────────────────────────────────────────────────
const video       = document.getElementById("video");
const canvas      = document.getElementById("canvas");
const ctx         = canvas.getContext("2d");
const statusPill  = document.getElementById("status-pill");
const statusText  = document.getElementById("status-text");
const errorBox    = document.getElementById("error-box");
const fpsBadge    = document.getElementById("fps-badge");

// ── State ──────────────────────────────────────────────────────────────────────
let landmarker     = null;
let lastVideoTime  = -1;
let lastPredictTs  = 0;
let fpsFrames      = 0;
let fpsLastTs      = performance.now();
const PREDICT_INTERVAL = 150; // ms between server calls

// ── Normalise landmarks (identical to the capture system) ─────────────────────
function normalizeHand(landmarks) {
  const { x: wx, y: wy, z: wz } = landmarks[0];
  const dx = landmarks[9].x - wx;
  const dy = landmarks[9].y - wy;
  const dz = landmarks[9].z - wz;
  const scale = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
  return landmarks.map(({ x, y, z }) => ({
    x: +((x - wx) / scale).toFixed(4),
    y: +((y - wy) / scale).toFixed(4),
    z: +((z - wz) / scale).toFixed(4),
  }));
}

// ── Draw hand skeleton on canvas ──────────────────────────────────────────────
function drawHand(results) {
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!results.landmarks?.length) return;

  for (const lm of results.landmarks) {
    // connections
    ctx.strokeStyle = "rgba(99,102,241,0.85)";
    ctx.lineWidth = 2.5;
    for (const [a, b] of CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(lm[a].x * canvas.width, lm[a].y * canvas.height);
      ctx.lineTo(lm[b].x * canvas.width, lm[b].y * canvas.height);
      ctx.stroke();
    }
    // joints
    for (let i = 0; i < lm.length; i++) {
      ctx.beginPath();
      ctx.arc(lm[i].x * canvas.width, lm[i].y * canvas.height, i === 0 ? 7 : 4.5, 0, Math.PI*2);
      ctx.fillStyle = i === 0 ? "#f59e0b" : "#22c55e";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

// ── Status pill colours ────────────────────────────────────────────────────────
function setStatus(state) {
  const colors = { loading: "#f59e0b", detecting: "#22c55e", no_hand: "#ef4444" };
  const labels = { loading: "Loading…", detecting: "Detecting", no_hand: "No hand detected" };
  statusPill.style.background = colors[state] ?? "#f59e0b";
  statusText.textContent = labels[state] ?? state;
}

// ── Confidence colour helper ───────────────────────────────────────────────────
function confColor(c) {
  if (c >= 0.80) return "#22c55e";
  if (c >= 0.55) return "#f59e0b";
  return "#ef4444";
}

// ── Update one model card ──────────────────────────────────────────────────────
function updateCard(prefix, result) {
  const { label, confidence, top3 } = result;
  const pct = (confidence * 100).toFixed(1);
  const color = prefix === "svm" ? "#a5b4fc" : "#86efac";
  const barColor = prefix === "svm" ? "#6366f1" : "#22c55e";

  // big sign
  const signEl = document.getElementById(`${prefix}-sign`);
  signEl.textContent = label;
  signEl.classList.remove("idle");
  signEl.style.color = confidence >= 0.5 ? color : "#64748b";

  // main bar
  document.getElementById(`${prefix}-bar`).style.width = `${confidence * 100}%`;
  document.getElementById(`${prefix}-bar`).style.background = barColor;

  // conf badge
  const badge = document.getElementById(`${prefix}-conf-badge`);
  badge.textContent = `${pct}%`;
  badge.style.background = confColor(confidence) + "22";
  badge.style.color = confColor(confidence);

  // top-3
  const top3El = document.getElementById(`${prefix}-top3`);
  top3El.innerHTML = "";
  const maxConf = top3[0][1];
  for (const [sign, conf] of top3) {
    const pctBar = maxConf > 0 ? (conf / maxConf) * 100 : 0;
    const row = document.createElement("div");
    row.className = "top3-row";
    row.innerHTML = `
      <span class="top3-sign" style="color:${color}">${sign}</span>
      <div class="top3-bar-wrap">
        <div class="top3-bar-fill" style="width:${pctBar}%;background:${barColor}88"></div>
      </div>
      <span class="top3-pct">${(conf * 100).toFixed(1)}%</span>`;
    top3El.appendChild(row);
  }
}

// ── Update consensus row ───────────────────────────────────────────────────────
function updateConsensus(svm, rf) {
  const box  = document.getElementById("consensus");
  const sign = document.getElementById("consensus-sign");
  const text = document.getElementById("consensus-text");

  if (svm.label === rf.label) {
    const avg = ((svm.confidence + rf.confidence) / 2 * 100).toFixed(1);
    box.className  = "consensus agree";
    sign.textContent = svm.label;
    sign.style.color = "#22c55e";
    sign.classList.remove("idle");
    text.textContent = `Both models agree · avg ${avg}%`;
  } else {
    box.className  = "consensus disagree";
    sign.textContent = `${svm.label} / ${rf.label}`;
    sign.style.color = "#f59e0b";
    sign.classList.remove("idle");
    text.textContent = `Models disagree — SVM: ${(svm.confidence*100).toFixed(1)}%  RF: ${(rf.confidence*100).toFixed(1)}%`;
  }
}

// ── Reset cards to idle ────────────────────────────────────────────────────────
function resetCards() {
  for (const prefix of ["svm", "rf"]) {
    const sign = document.getElementById(`${prefix}-sign`);
    sign.textContent = "—";
    sign.classList.add("idle");
    sign.style.color = "";
    document.getElementById(`${prefix}-bar`).style.width = "0%";
    document.getElementById(`${prefix}-top3`).innerHTML = "";
    const badge = document.getElementById(`${prefix}-conf-badge`);
    badge.textContent = "—";
    badge.style.background = "#1e293b";
    badge.style.color = "#475569";
  }
  const box = document.getElementById("consensus");
  box.className = "consensus";
  document.getElementById("consensus-sign").textContent = "—";
  document.getElementById("consensus-sign").classList.add("idle");
  document.getElementById("consensus-sign").style.color = "";
  document.getElementById("consensus-text").textContent = "waiting for hand…";
  document.getElementById("card-svm").classList.remove("agreement");
  document.getElementById("card-rf").classList.remove("agreement");
}

// ── Send landmarks to server, update UI ───────────────────────────────────────
let predictInFlight = false;
async function sendPredict(normalized) {
  if (predictInFlight) return;
  predictInFlight = true;
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ landmarks: normalized }),
    });
    if (!res.ok) return;
    const data = await res.json();
    if (data.error) return;

    updateCard("svm", data.svm);
    updateCard("rf",  data.rf);
    updateConsensus(data.svm, data.rf);

    // highlight cards on agreement
    const agree = data.svm.label === data.rf.label;
    document.getElementById("card-svm").classList.toggle("agreement", agree);
    document.getElementById("card-rf").classList.toggle("agreement", agree);
  } catch {
    // network error — just skip this frame
  } finally {
    predictInFlight = false;
  }
}

// ── Detection rAF loop ─────────────────────────────────────────────────────────
function detectFrame() {
  if (!landmarker) { requestAnimationFrame(detectFrame); return; }
  if (video.readyState < 2) { requestAnimationFrame(detectFrame); return; }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const results = landmarker.detectForVideo(video, video.currentTime * 1000);
    const hasHand = !!results.landmarks?.length;

    drawHand(results);
    setStatus(hasHand ? "detecting" : "no_hand");

    if (hasHand) {
      const now = performance.now();
      if (now - lastPredictTs > PREDICT_INTERVAL) {
        lastPredictTs = now;
        sendPredict(normalizeHand(results.landmarks[0]));
      }
    } else {
      resetCards();
    }

    // FPS counter
    fpsFrames++;
    const elapsed = performance.now() - fpsLastTs;
    if (elapsed >= 1000) {
      fpsBadge.textContent = `${Math.round(fpsFrames * 1000 / elapsed)} fps`;
      fpsFrames = 0;
      fpsLastTs = performance.now();
    }
  }

  requestAnimationFrame(detectFrame);
}

// ── Init ───────────────────────────────────────────────────────────────────────
async function init() {
  setStatus("loading");
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
    });

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: 1280, height: 720 },
    });
    video.srcObject = stream;
    await video.play();

    setStatus("no_hand");
    requestAnimationFrame(detectFrame);
  } catch (err) {
    errorBox.style.display = "";
    errorBox.textContent = "Error: " + err.message;
    setStatus("loading");
  }
}

init();
</script>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    print(f"Open http://localhost:{args.port} in your browser")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
