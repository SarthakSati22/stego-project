"""Risk-weighted split of the secret across audio, video and image.

For every clip we hide small test payloads of growing size (LADDER bits) in each carrier on its
own and ask that carrier's detector how suspicious the result looks (0 = looks clean, 1 = certainly
stego). The size at which the risk crosses RISK_LIMIT is the carrier's SAFE CAPACITY for this clip.
The secret is split in proportion to the safe capacities: a carrier the detector flags early gets a
small piece, a carrier that stays unflagged gets a large one. There is no fixed starting split; if all
three carriers have the same safe capacity the pieces are equal. Only a tiny FLOOR (every carrier must
hold at least a few bytes) and a CAP (no carrier holds nearly everything) keep the pieces usable.
"""
import pickle
from pathlib import Path

import numpy as np

from src.audio import stft_embed as A
from src.crossmodal import image_carrier as I
from src.crossmodal.image_features import image_features
from src.crossmodal.pipeline3 import KEYS
from src.steganalysis.features import audio_features, video_features
from src.video import dct_embed as VS
from src.video import motion_embed as VM

PRIOR = {"audio": 1.0, "video": 1.0, "image": 1.0}   # equal weights: no built-in preference
LADDER = (16, 32, 64, 128, 256)          # test payload sizes in bits
RISK_LIMIT = 0.5                         # a carrier counts as "caught" above this detector score
MAX_SAFE = 512                           # safe capacity when even the largest test stays unflagged
MIN_SAFE = 8                             # ... and when even the smallest is flagged
FLOOR, CAP = 0.05, 0.80
STRENGTH = 1.0


def bounded_shares(w, lo=FLOOR, hi=CAP):
    """Shares proportional to w, adding up to 1, with every share inside [lo, hi].

    Solves  sum(clip(lam * w, lo, hi)) = 1  for lam by bisection (the sum only grows with lam).
    """
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    n = len(w)
    if not (n * lo <= 1.0 <= n * hi):
        return w
    low, high = 0.0, 1.0
    while np.clip(high * w, lo, hi).sum() < 1.0:
        high *= 2
    for _ in range(100):
        mid = 0.5 * (low + high)
        if np.clip(mid * w, lo, hi).sum() < 1.0:
            low = mid
        else:
            high = mid
    c = np.clip(0.5 * (low + high) * w, lo, hi)
    return c / c.sum()


def safe_capacity(sizes, risks, limit=RISK_LIMIT) -> float:
    """Bits a carrier can hold before its detector score crosses the limit (log-linear interpolation)."""
    sizes = np.asarray(sizes, dtype=float)
    risks = np.asarray(risks, dtype=float)
    if risks[0] >= limit:
        return float(MIN_SAFE)
    for i in range(1, len(sizes)):
        if risks[i] >= limit:
            t = (limit - risks[i - 1]) / max(risks[i] - risks[i - 1], 1e-9)
            return float(np.exp(np.log(sizes[i - 1]) + t * (np.log(sizes[i]) - np.log(sizes[i - 1]))))
    return float(MAX_SAFE)


def shares_from_safe(safe: dict, prior=PRIOR, strength=STRENGTH) -> dict:
    """safe: {carrier: safe capacity in bits}. Returns {carrier: share}, adding up to 1."""
    names = list(prior)
    w = [prior[n] * max(float(safe[n]), 1.0) ** strength for n in names]
    return dict(zip(names, (float(x) for x in bounded_shares(w))))


def load_detectors(det_path="results/detectors.pkl", img_path="results/image_detector.pkl") -> dict:
    with open(det_path, "rb") as f:
        d = pickle.load(f)
    img = None
    if Path(img_path).exists():
        with open(img_path, "rb") as f:
            img = pickle.load(f)
    return {"audio": d["audio"], "video": d["video"], "image": img}


def probe_curves(audio, frames, still, dets: dict, video="standard", sizes=LADDER, seed=0) -> dict:
    """{carrier: [detector risk at each test size]} with only that carrier holding the test payload."""
    rng = np.random.default_rng(seed)
    vid = VM if video == "motion" else VS
    curves = {"audio": [], "video": [], "image": []}
    for nb in sizes:
        bits = rng.integers(0, 2, nb).astype(np.uint8)
        sa = A.embed(audio, bits, KEYS["audio"])
        curves["audio"].append(float(dets["audio"].score(np.asarray(audio_features(sa))[None, :])[0]))
        sv = vid.embed(frames, bits, KEYS["video"])
        curves["video"].append(float(dets["video"].score(np.asarray(video_features(sv))[None, :])[0]))
        if dets.get("image") is not None:
            si = I.embed(still, bits, KEYS["image"])
            curves["image"].append(float(dets["image"].score(np.asarray(image_features(si))[None, :])[0]))
    if dets.get("image") is None:
        curves["image"] = None            # no image detector trained
    return curves


def risk_split(audio, frames, still, dets: dict, video="standard") -> dict:
    """Everything the API needs: shares, safe capacities and the risk of each carrier at 128 bits."""
    curves = probe_curves(audio, frames, still, dets, video)
    safe = {c: (safe_capacity(LADDER, curves[c]) if curves[c] is not None else float(MAX_SAFE))
            for c in ("audio", "video", "image")}
    at = LADDER.index(128)
    return {"shares": shares_from_safe(safe), "safe_bits": safe,
            "risk_at_128": {c: (curves[c][at] if curves[c] is not None else None) for c in curves},
            "image_detector": curves["image"] is not None}
