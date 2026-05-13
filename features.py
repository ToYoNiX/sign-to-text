import numpy as np

# (MCP_index, PIP_index, TIP_index) for each finger
FINGERS = [
    (1, 2, 4),  # thumb:  CMC, MCP, Tip
    (5, 6, 8),  # index:  MCP, PIP, Tip
    (9, 10, 12),  # middle: MCP, PIP, Tip
    (13, 14, 16),  # ring:   MCP, PIP, Tip
    (17, 18, 20),  # pinky:  MCP, PIP, Tip
]

N_FEATURES = 86  # change nnumber from 63 to 86


def extract(landmarks) -> np.ndarray:

    if isinstance(landmarks, np.ndarray):
        lm = landmarks.astype(np.float32)
    else:
        lm = np.array([[p["x"], p["y"], p["z"]] for p in landmarks], dtype=np.float32)  # (21, 3)

    raw = lm.flatten()

    directions: list[float] = []
    curls: list[float] = []

    for mcp_i, pip_i, tip_i in FINGERS:
        mcp = lm[mcp_i]
        pip = lm[pip_i]
        tip = lm[tip_i]

        d_vec = tip - mcp
        n = np.linalg.norm(d_vec)
        directions.extend((d_vec / n).tolist() if n > 1e-6 else [0.0, 0.0, 0.0])

        # Curl = 1 − cos(angle at PIP)   0 = straight, ~2 = fully bent
        v1 = mcp - pip
        n1 = np.linalg.norm(v1)
        v2 = tip - pip
        n2 = np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos_a = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
            curls.append(1.0 - cos_a)
        else:
            curls.append(0.0)

    v_index = lm[5] - lm[0]
    v_pinky = lm[17] - lm[0]
    normal = np.cross(v_index, v_pinky)
    n_norm = np.linalg.norm(normal)
    palm_n = (normal / n_norm).tolist() if n_norm > 1e-6 else [0.0, 0.0, 0.0]

    return np.array(raw.tolist() + directions + curls + palm_n, dtype=np.float32)


def extract_from_dict(d: dict) -> np.ndarray:
    return extract(d["frames"][0]["landmarks"])


def recompute_from_raw63(flat63: np.ndarray) -> np.ndarray:

    return extract(flat63.reshape(21, 3))
