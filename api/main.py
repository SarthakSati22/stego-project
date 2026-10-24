"""Backend: hide, extract and score a secret across the audio and video of a clip."""
import json
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.channel.attacks import apply_attack
from src.crypto.secret import derive_key, encode_secret
from src.eval.metrics import psnr, snr, ssim
from src.media_io import read_luma, read_wav, write_luma, write_wav
from src.pipeline import embed_clip, recover_secret

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
DETECTORS = ROOT / "results" / "detectors.pkl"
OUT = ROOT / "api_out"
OUT.mkdir(exist_ok=True)

STRATEGY_SHARE = {"audio_only": 1.0, "video_only": 0.0, "fixed_50_50": 0.5}
CHANNELS = ["none", "mp3_128", "mp3_64", "h264_crf23", "h264_crf28", "h265_crf23",
            "h265_crf28", "drop_5", "drop_10", "resize_75",
            "mp3_128+h264_crf23", "mp3_64+h264_crf28"]

app = FastAPI(title="Cross-Modal Stego API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@lru_cache(maxsize=1)
def manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST).set_index("clip_id")


@lru_cache(maxsize=1)
def detector():
    """The allocator's trained detectors, or None if they are not available."""
    if not DETECTORS.exists():
        return None
    try:
        from src.allocator.detect import load_detect_fn
        return load_detect_fn(str(DETECTORS))
    except Exception as e:      # the API still works without detection scores
        print(f"[api] detectors not loaded: {e}")
        return None


def load_clip(clip_id: str):
    if clip_id not in manifest().index:
        raise HTTPException(404, f"unknown clip_id: {clip_id}")
    row = manifest().loc[clip_id]
    return read_wav(ROOT / row["audio_path"]), read_luma(str(ROOT / row["video_path"]))


def score(audio, frames):
    fn = detector()
    if fn is None:
        return None
    p_a, p_v, p_f = fn(audio, frames)
    return {"audio": float(p_a), "video": float(p_v), "fused": float(p_f)}


def through_channel(name, audio, frames, seed=0):
    rng = np.random.default_rng(seed)
    for part in name.split("+"):
        audio, frames = apply_attack(part, audio, frames, rng)
    return audio, frames


def clean(x):
    """JSON cannot hold inf/nan."""
    return float(x) if np.isfinite(x) else None


class EmbedRequest(BaseModel):
    clip_id: str
    secret: str = Field(min_length=1, max_length=128)
    passphrase: str = Field(min_length=1)
    strategy: str = "fixed_50_50"
    share_audio: float | None = Field(default=None, ge=0, le=1)


class ExtractRequest(BaseModel):
    job_id: str
    passphrase: str
    channel: str = "none"


@app.get("/health")
def health():
    return {"status": "ok", "detectors": DETECTORS.exists()}


@app.get("/clips")
def clips(split: str = "test", limit: int = 20):
    df = manifest().reset_index()
    df = df[df["split"] == split].head(limit)
    return df[["clip_id", "actor", "emotion", "split"]].to_dict("records")


@app.get("/options")
def options():
    return {"strategies": list(STRATEGY_SHARE), "channels": CHANNELS}


@app.post("/embed")
def embed(req: EmbedRequest):
    share = req.share_audio
    if share is None:
        if req.strategy not in STRATEGY_SHARE:
            raise HTTPException(400, f"strategy must be one of {list(STRATEGY_SHARE)}")
        share = STRATEGY_SHARE[req.strategy]
    audio, frames = load_clip(req.clip_id)
    bits = encode_secret(req.secret.encode(), derive_key(req.passphrase))
    try:
        sa, sv, n_a, n_v = embed_clip(audio, frames, bits, share)
    except Exception as e:      # usually: payload too large for this carrier
        raise HTTPException(400, f"cannot embed: {e}")
    job = uuid.uuid4().hex[:12]
    folder = OUT / job
    folder.mkdir()
    write_wav(folder / "stego.wav", sa)
    write_luma(str(folder / "stego.mkv"), sv)
    (folder / "meta.json").write_text(json.dumps(
        {"clip_id": req.clip_id, "n_a": n_a, "n_v": n_v, "share_audio": share}))
    return {
        "job_id": job, "bits": int(len(bits)), "bits_audio": n_a, "bits_video": n_v,
        "share_audio": share,
        "quality": {"snr_db": clean(snr(audio, sa)) if n_a else None,
                    "psnr_db": clean(psnr(frames, sv)) if n_v else None,
                    "ssim": clean(ssim(frames, sv)) if n_v else None},
        "detection_clean": score(audio, frames),
        "detection_stego": score(sa, sv),
    }


@app.post("/extract")
def extract(req: ExtractRequest):
    folder = OUT / req.job_id
    if not (folder / "meta.json").exists():
        raise HTTPException(404, "unknown job_id")
    if req.channel not in CHANNELS:
        raise HTTPException(400, f"channel must be one of {CHANNELS}")
    meta = json.loads((folder / "meta.json").read_text())
    audio = read_wav(folder / "stego.wav")
    frames = read_luma(str(folder / "stego.mkv"))
    audio, frames = through_channel(req.channel, audio, frames)
    secret = recover_secret(audio, frames, meta["n_a"], meta["n_v"],
                            derive_key(req.passphrase))
    return {"recovered": secret is not None, "channel": req.channel,
            "secret": secret.decode(errors="replace") if secret is not None else None}


@app.get("/download/{job_id}/{kind}")
def download(job_id: str, kind: str):
    names = {"audio": "stego.wav", "video": "stego.mkv"}
    if kind not in names or not (OUT / job_id / names[kind]).exists():
        raise HTTPException(404, "not found")
    return FileResponse(OUT / job_id / names[kind], filename=f"{job_id}_{names[kind]}")