# ARSL Sign Recognition

Arabic Sign Language (ARSL) static sign recognition using MediaPipe hand landmarks and classical ML classifiers (SVM + Random Forest).

MediaPipe runs entirely in the browser — only 63 normalized floats per frame are ever sent to the classifier, making the system lightweight, lighting-invariant, and background-invariant.

---

## How it works

```
Camera → MediaPipe Hands (browser, WebAssembly)
       → 21 landmarks × (x, y, z)
       → normalize (wrist origin, palm-size scale)
       → SVM / Random Forest → Arabic letter
```

**Normalization** removes position and size variation before classification:
1. Subtract wrist (landmark 0) from all points → hand always at origin
2. Divide by wrist→middle-MCP distance → removes hand size differences between people

---

## Project layout

```
raw/                  captured samples (JSON, one file per capture)
dataset/              filtered + mirrored samples ready for training (gitignored)
models/               trained model files (gitignored — regenerate with train.py)
templates/
  index.html          unified browser UI (served in two modes via CONFIG injection)
  style.css           stylesheet
model.onnx            exported SVM pipeline — committed, used by the static site
label_map.json        index → Arabic label — committed alongside model.onnx
build_dataset.py      quality filtering + mirroring pipeline
train.py              augmentation + SVM/RF training
export_onnx.py        export SVM to ONNX (writes model.onnx + label_map.json to root)
build.py              assemble _site/ for GitHub Pages deployment
serve.py              local dev server — serves UI + POST /predict + WS /predict (auth)
predict.py            headless prediction API or interactive terminal
requirements.txt      pip dependencies
pyproject.toml        Poetry dependencies (package-mode = false)
poetry.lock           locked dependency versions
auth.json             generated on first run — gitignored, holds the API token
tests/
  test_predict.py     unit tests for the prediction pipeline
.github/workflows/
  deploy.yml          on push to main — builds and deploys to GitHub Pages
  lint.yml            on pull request — ruff check + format
  test.yml            on push/PR — pytest
```

---

## Quickstart

**Install dependencies**

With pip:
```bash
pip install -r requirements.txt
```

With Poetry:
```bash
poetry install
```

**1. Build dataset** (filter raw captures, generate mirrors)
```bash
python build_dataset.py          # keeps best 30 samples per sign
python build_dataset.py --keep 50

poetry run python build_dataset.py
poetry run python build_dataset.py --keep 50
```

**2. Train**
```bash
python train.py                  # 15 augmented copies per sample (default)
python train.py --augment 25
python train.py --no-augment     # raw data only

poetry run python train.py
poetry run python train.py --augment 25
poetry run python train.py --no-augment
```
Outputs `models/svm.pkl`, `models/rf.pkl`, `models/scaler.pkl`, `models/label_map.json`.

**3. Run the live demo**
```bash
python serve.py                  # http://localhost:8080
python serve.py --port 9000

poetry run python serve.py
poetry run python serve.py --port 9000
```
Open the URL in a browser. Allow camera access. Show your hand.

---

## Prediction API

`serve.py` exposes two transports on the same `/predict` route — HTTP for one-off requests, WebSocket for real-time streaming.

**Start the server**
```bash
python serve.py          # http://localhost:8080
python serve.py --port 9000

poetry run python serve.py
poetry run python serve.py --port 9000
```

**WebSocket — real-time streaming**

Connect once and send one message per video frame. No per-frame HTTP handshake overhead.

Requires a token (see Auth below):
```
ws://localhost:8080/predict?token=<your-token>
```

Send:
```json
{ "landmarks": [{"x": 0.0, "y": 0.0, "z": 0.0}, "... 21 points total"] }
```

Receive:
```json
{
  "svm": { "label": "ح", "confidence": 0.97, "top3": [["ح", 0.97], ["خ", 0.02], ["ه", 0.01]] },
  "rf":  { "label": "ح", "confidence": 0.94, "top3": [["ح", 0.94], ["خ", 0.05], ["ه", 0.01]] }
}
```

If the token is missing or wrong the connection is closed immediately with code 1008.

**HTTP — one-off requests**

No token required (used by the browser UI).

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"landmarks": [{"x":0,"y":0,"z":0}, ...]}'   # 21 points
```

Response shape is identical to the WebSocket response above.

CORS is enabled for all origins.

---

## Auth

On first run `serve.py` generates a random token and writes it to `auth.json` next to the script:

```
====================================================
  Auth token generated — auth.json
  Token: <your-token>
  Delete auth.json to regenerate.
====================================================
```

Subsequent runs load the token silently.

**Get the token at any time**
```bash
cat auth.json
# { "token": "abc123..." }

# extract just the value
python -c "import json; print(json.load(open('auth.json'))['token'])"
poetry run python -c "import json; print(json.load(open('auth.json'))['token'])"
```

**Rotate the token** — delete `auth.json` and restart the server. A new token will be generated and printed.

`auth.json` is gitignored and never committed.

---

## Headless API / terminal

`predict.py` exposes the models without a browser UI.

**API server** (no HTML, just JSON endpoints)
```bash
python predict.py                          # http://localhost:8000
python predict.py --port 9000 --model rf

poetry run python predict.py
poetry run python predict.py --port 9000 --model rf

# POST /predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"landmarks": [{"x":0,"y":0,"z":0}, ...]}'   # 21 points

# GET /health
curl http://localhost:8000/health
```

**Interactive terminal**
```bash
python predict.py -i             # paste landmark JSON, press Enter
python predict.py -i --model rf

poetry run python predict.py -i
poetry run python predict.py -i --model rf
```

---

## CI

| Workflow | Trigger | What it does |
|---|---|---|
| `lint.yml` | Pull request | `ruff check` + `ruff format --check` |
| `test.yml` | Push to main, pull request | `pytest tests/ -v` |
| `deploy.yml` | Push to main (model/template changes) | Builds `_site/` and deploys to GitHub Pages |

Run locally:
```bash
ruff check . && ruff format --check .
pytest tests/ -v

# or with Poetry
poetry run ruff check . && poetry run ruff format --check .
poetry run pytest tests/ -v
```

---

## Deploying to GitHub Pages

The static site is a fully offline browser app — no server needed. It runs MediaPipe and the SVM (via ONNX) entirely in the browser.

`build.py` assembles `_site/` from `templates/index.html`, `templates/style.css`, `model.onnx`, and `label_map.json`. GitHub Actions runs this automatically on every push that touches those files.

### First-time setup

**1. Enable GitHub Actions as the Pages source**

Go to your repo on GitHub → **Settings** → **Pages** → under *Build and deployment*, set **Source** to **GitHub Actions** (not "Deploy from a branch").

That's it. The workflow file is already in the repo at [.github/workflows/deploy.yml](.github/workflows/deploy.yml). The next qualifying push will trigger it.

**2. Check the deployment**

Go to **Actions** tab in your repo. You'll see a *Deploy to GitHub Pages* workflow run. When it goes green, your site is live at `https://<your-username>.github.io/<repo-name>/`.

### Updating the model after retraining

```bash
python train.py 2>&1 | grep -E "CV accuracy|Saved|Training|Loading" && python export_onnx.py
# or
poetry run python train.py 2>&1 | grep -E "CV accuracy|Saved|Training|Loading" && poetry run python export_onnx.py

git add model.onnx label_map.json
git commit -m "update model"
git push                                 # Actions picks it up and redeploys
```

### Testing the build locally before pushing

```bash
python build.py          # assembles _site/
poetry run python build.py

# open _site/index.html in a browser — this is exactly what gets deployed
```

---

## Raw data format

Each file in `raw/` is a JSON capture from the browser capture tool:

```json
{
  "label": "ح",
  "type": "static",
  "mirrorable": true,
  "captured_at": "2026-05-07T21:46:15.036Z",
  "frame_count": 1,
  "frames": [
    {
      "landmarks": [
        { "x": 0, "y": 0, "z": 0 },
        { "x": 0.5671, "y": -0.1327, "z": 0.4982 },
        "... 19 more points"
      ]
    }
  ]
}
```

Landmarks are already normalized at capture time (wrist at origin, scaled by palm size). `mirrorable: true` means `build_dataset.py` will generate a left-hand mirror copy automatically.

---

## Augmentation

`train.py` generates 15 augmented copies per real sample:

| Transform | Probability | What it simulates |
|---|---|---|
| Gaussian noise | 60% | Hand tremor, sensor jitter |
| 3D rotation | 50% | ±20° tilt/yaw, ±15° twist |
| Dorsal flip | 40% | Back-of-hand view (with occlusion noise on hidden landmarks) |
| Per-finger scale | 100% | Different finger length proportions between people |

The dorsal flip adds extra noise to the 10 landmarks that would be occluded when viewing from the back (fingertips + DIP/IP joints), matching how MediaPipe estimates hidden points.
