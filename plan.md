# ARSL (Arabic Sign Language) Translation System — Project Plan

## Overview

A two-stage pipeline that separates **feature extraction** from **classification**. A pre-trained computer vision model runs locally in the browser to extract hand landmarks, and only the landmark data is sent to a backend classifier. This avoids the pitfalls of end-to-end video models and makes the system lightweight, private, and generalizable.

---

## Architecture

```
Camera → MediaPipe Hands (browser, local) → 21 Landmarks × (x, y, z)
                                                        ↓
                                              Normalize + Buffer
                                                        ↓
                                            Backend API (your server)
                                                        ↓
                                         LSTM / Classifier → Sign Label
```

### Why This Approach

| Traditional Video Model | This Approach |
|---|---|
| Raw pixels as input (huge) | 63 numbers per frame (tiny) |
| Thousands of videos needed | Hundreds of samples sufficient |
| Sensitive to lighting and background | Invariant to both |
| Hard to debug failures | Fully interpretable landmark data |
| Needs GPU for inference | Runs on any device |

---

## Stage 1 — Frontend (Browser)

### Technology

**MediaPipe Tasks Vision** (`@mediapipe/tasks-vision`) — runs entirely in the browser via WebAssembly. No external API calls during inference. Camera feed never leaves the device.

- Downloads WASM runtime (~2MB) and model file (~8MB) once on first load, then browser-cached
- Runs at 30+ FPS on most devices using WebGL GPU acceleration
- Outputs 21 3D landmarks per hand, handedness, and palm orientation

### What Gets Extracted Per Frame

Each hand produces **21 landmarks**, each with:
- `x` — horizontal position (0 to 1, normalized to frame width)
- `y` — vertical position (0 to 1, normalized to frame height)
- `z` — depth relative to wrist (negative = closer to camera)

Total: **63 floats per hand per frame**

### Running Mode

Use `VIDEO` mode (not `IMAGE`) for live camera streams. Requires a timestamp on every `detectForVideo()` call so MediaPipe can track motion between frames.

### What Gets Sent to Backend

- **Static signs** — single normalized frame (63 numbers)
- **Dynamic signs** — sequence of 30 normalized frames (1890 numbers)

---

## Stage 2 — Backend (Classifier)

### Model Choice by Sign Type

| Sign Type | Example | Recommended Model |
|---|---|---|
| Static (shape only) | Letters, static words | SVM, Random Forest, small MLP |
| Dynamic (movement) | "Hello", "Thank you" | LSTM or small Transformer |

### Input to the Model

Normalized landmark sequences — not raw coordinates. See normalization section below.

### Tech Stack Suggestion

- **Python** backend (FastAPI or Flask)
- **scikit-learn** for static classifiers
- **PyTorch** for LSTM on dynamic signs
- Serve both under a single `/predict` endpoint that accepts a sequence and returns a label + confidence

---

## Data Collection

### Sample Structure

Each recorded sample is stored as JSON:

```json
{
  "label": "مرحبا",
  "type": "dynamic",
  "mirrorable": false,
  "frames": [
    {
      "timestamp": 0,
      "landmarks": [
        { "x": 0.45, "y": 0.60, "z": -0.08 },
        "... 20 more points"
      ]
    },
    "... more frames"
  ]
}
```

### Sign Types

- **Static signs** — 1 frame captured (or average of a short 10-frame window to reduce noise)
- **Dynamic signs** — sequence of frames captured from button hold to button release, then trimmed and resampled

### Capture Flow

```
User presses HOLD
    → Start buffering frames
User presses RELEASE
    → Trim leading/trailing frames where no hand detected
    → Normalize the sequence
    → Resample to fixed length (30 frames)
    → Save to dataset
```

### Recommended Samples Per Sign (Before Augmentation)

- Static signs: **30–50 real samples**
- Dynamic signs: **30–50 real samples**

After augmentation this becomes 450–1000+ effective training samples per sign.

### File Structure

```
dataset/
  raw/
    مرحبا_001.json
    مرحبا_002.json
    شكرا_001.json
  processed/
    X_train.npy      ← flattened normalized sequences
    y_train.npy      ← label indices
    label_map.json   ← index → Arabic label
```

---

## Normalization (Critical)

Apply before saving any sample. This removes variation caused by hand position in frame and hand size between people — the two biggest sources of generalization failure.

### Step 1 — Translate to Wrist Origin

Subtract the wrist landmark (point 0) from all other points so the hand is always centered at the origin regardless of where in the frame it appears.

```python
landmarks = landmarks - landmarks[0]  # wrist becomes (0, 0, 0)
```

### Step 2 — Scale by Palm Size

Divide all points by the distance between the wrist (point 0) and the middle finger MCP joint (point 9). This removes hand size variation between different people.

```python
palm_size = np.linalg.norm(landmarks[9] - landmarks[0])
landmarks = landmarks / palm_size
```

### Step 3 — Resample to Fixed Length (Dynamic Signs)

Resample every recording to exactly 30 frames using interpolation. This handles speed variation — a fast signer and a slow signer both produce identical-length input to the model.

```python
from scipy.interpolate import interp1d
import numpy as np

def resample_sequence(sequence, target_frames=30):
    x_old = np.linspace(0, 1, len(sequence))
    x_new = np.linspace(0, 1, target_frames)
    f = interp1d(x_old, sequence, axis=0)
    return f(x_new)
```

---

## Data Augmentation

Apply after normalization, before training. One real sample generates 15–20 augmented copies, dramatically reducing the number of real recordings needed.

### Augmentations

**Mirror (flip hand)**
Negate the x-axis to simulate the opposite hand signing the same sign.
```python
augmented[:, :, 0] *= -1
```
Only apply if `mirrorable: true` for that sign (see ARSL note below).

**Add noise**
Small random perturbations simulate natural hand tremor and sensor jitter.
```python
noise = np.random.normal(0, 0.01, landmarks.shape)
augmented = landmarks + noise
```
Keep scale between 0.005 and 0.02 — too high corrupts the sign shape.

**Scale**
Simulate residual hand size differences after normalization.
```python
scale = np.random.uniform(0.85, 1.15)
augmented = landmarks * scale
```

**Rotate (2D around wrist)**
Simulate slight tilt or angle differences in how the user holds their hand.
```python
angle = np.radians(np.random.uniform(-15, 15))
cos, sin = np.cos(angle), np.sin(angle)
augmented[:, :, 0] = landmarks[:, :, 0] * cos - landmarks[:, :, 1] * sin
augmented[:, :, 1] = landmarks[:, :, 0] * sin + landmarks[:, :, 1] * cos
```
Keep range to ±15° to avoid physically impossible hand poses.

**Speed variation**
Resample at 0.8× and 1.2× speed before applying the fixed-length resample. Teaches the model that the same sign at different speeds is still the same sign.

### What NOT to Augment

- Do not rotate 3D aggressively — can create impossible hand poses
- Do not mirror direction-dependent dynamic signs
- Do not add heavy noise to fingertip landmarks — they carry the most discriminative information for similar-looking signs

### Augmentation Pipeline

```python
def augment_sample(landmarks, mirrorable=False):
    if mirrorable and np.random.rand() > 0.5:
        landmarks = mirror(landmarks)
    if np.random.rand() > 0.5:
        landmarks = add_noise(landmarks)
    if np.random.rand() > 0.5:
        landmarks = rotate(landmarks)
    landmarks = scale(landmarks)  # always apply
    return landmarks

augmented_dataset = []
for sample in real_samples:
    for _ in range(15):
        augmented_dataset.append(
            augment_sample(sample['landmarks'], sample['mirrorable'])
        )
```

---

## ARSL-Specific Notes

### On Mirroring

Each sign in the dataset has a `mirrorable` flag. The default is `false` — only set to `true` after verifying with a native ARSL signer.

**Safe to mirror:** Most static hand shapes where position relative to the body does not change meaning.

**Do not mirror:**
- Signs influenced by Arabic writing direction (right-to-left directionality)
- Signs with directional movement where direction carries meaning
- Two-handed signs where both hands have asymmetric roles

**Action item:** Go through the sign list with a native ARSL signer or linguist and explicitly mark `mirrorable: true` for verified signs before running augmentation.

### On Dynamic Signs

ARSL has signs composed of movement, not just hand shape. These require the LSTM path. Identify which signs in your target vocabulary are dynamic during data collection and tag them with `"type": "dynamic"` so the pipeline routes them correctly.

---

## Development Phases

### Phase 1 — Proof of Concept (Now)
- [ ] Build simple React frontend with MediaPipe Hands
- [ ] Display live landmark overlay on camera feed
- [ ] Display raw landmark JSON to verify data quality
- [ ] Confirm 30+ FPS landmark extraction works on target devices

### Phase 2 — Data Collection Tool
- [ ] Build recording UI with hold-to-capture interaction
- [ ] Implement live landmark normalization (translate + scale)
- [ ] Add trimming of no-hand frames at start/end
- [ ] Export captured samples as JSON
- [ ] Build label management (sign list, mirrorable flag, type flag)

### Phase 3 — Data Pipeline
- [ ] Write normalization script for collected JSON files
- [ ] Write resampling script (fixed 30 frames)
- [ ] Write augmentation pipeline (noise, rotate, scale, mirror)
- [ ] Convert to numpy arrays for training (X_train, y_train)

### Phase 4 — Model Training
- [ ] Train static classifier (SVM or MLP) on static signs
- [ ] Train LSTM on dynamic sign sequences
- [ ] Evaluate per-sign accuracy and identify weak signs
- [ ] Collect more real samples for weak signs only

### Phase 5 — Backend API
- [ ] Wrap trained models in FastAPI endpoint
- [ ] Accept landmark sequence, return label + confidence
- [ ] Connect React frontend to backend

### Phase 6 — Iteration
- [ ] Collect data from multiple signers for generalization
- [ ] Test on left-handed signers
- [ ] Expand vocabulary

---

## Key Numbers to Remember

| Parameter | Value | Reason |
|---|---|---|
| Landmarks per hand | 21 | MediaPipe fixed output |
| Coordinates per landmark | 3 (x, y, z) | |
| Floats per frame | 63 | 21 × 3 |
| Target sequence length | 30 frames | Fixed-length LSTM input |
| Floats per dynamic sample | 1890 | 30 × 63 |
| Noise scale | 0.005 – 0.02 | Safe augmentation range |
| Rotation range | ±15° | Beyond this risks impossible poses |
| Scale range | 0.85 – 1.15 | |
| Augmentation multiplier | ~15× | Per real sample |
| Real samples needed (target) | 30–50 per sign | Before augmentation |
