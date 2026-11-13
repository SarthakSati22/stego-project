"""Three-carrier endpoints (audio + video + image) with k-of-n erasure coding."""
import json
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api import db
from api.db import history as _history
from api.main import OUT, ROOT, clean, load_clip
from src.channel import attacks as T
from src.crossmodal import image_carrier as I
from src.crossmodal.pipeline3 import PLANS, embed, extract
from src.crossmodal.riskshare import load_detectors, risk_split
from src.crypto.secret import derive_key
from src.eval.metrics import psnr, snr, ssim
from src.media_io import read_luma, read_wav, write_luma, write_wav

router = APIRouter()

MAX_SECRET_BYTES = 96            # keeps every fragment inside the smallest carrier (image/motion video)
PLAN_INFO = {
    "audio_only": "Everything in the audio only. Fails if the audio is destroyed.",
    "video_only": "Everything in the video only. Fails if the video is destroyed.",
    "image_only": "Everything in the image only. Fails if the image is destroyed.",
    "split3": "Cut in 3 pieces, one per carrier. Needs ALL three: the weakest plan.",
    "erasure_2of3": "3 smart pieces, any 2 rebuild the secret. Survives one destroyed carrier.",
    "erasure_2of3_motion": "Same as 2-of-3, with the motion-aware video embedder.",
    "replicate_3": "A full copy in every carrier. Survives two destroyed carriers (largest cost).",
    "video_only_motion": "Video only, motion-aware embedder.",
    "risk_split3": "Cut in 3 pieces whose sizes follow the detector risk measured on THIS clip: "
                   "the earlier a carrier gets flagged, the smaller its piece. Needs ALL three.",
    "risk_split3_motion": "Risk-weighted split (see above) with the motion-aware video embedder.",
}
# (audio attack, video attack, jpeg quality). "kill_*" settings are deliberately extreme.
SCENARIOS = {
    "clean": (None, None, None),
    "mild": ("mp3_128", "h264_crf23", 75),
    "medium": ("mp3_64", "h264_crf28", 30),
    "kill_audio": ("mp3_32", None, None),
    "kill_video": (None, "h265_crf35", None),
    "kill_image": (None, None, 5),
    "kill_audio+video": ("mp3_32", "h265_crf35", None),
    "kill_audio+image": ("mp3_32", None, 5),
    "kill_video+image": (None, "h265_crf35", 5),
}


class Embed3Request(BaseModel):
    clip_id: str
    secret: str = Field(min_length=1, max_length=96)
    passphrase: str = Field(min_length=1)
    plan: str = "erasure_2of3"


class Extract3Request(BaseModel):
    job_id: str
    passphrase: str
    scenario: str = "clean"


_DETS = {}


def detectors():
    """Trained detectors (audio, video, and the image one if it has been trained), loaded once."""
    if "d" not in _DETS:
        try:
            _DETS["d"] = load_detectors(str(ROOT / "results" / "detectors.pkl"),
                                        str(ROOT / "results" / "image_detector.pkl"))
        except Exception as e:
            raise HTTPException(400, f"risk-weighted split needs results/detectors.pkl ({e})")
    return _DETS["d"]


def through_scenario(name, audio, frames, still):
    a, v, q = SCENARIOS[name]
    if a:
        audio = T.mp3(audio, int(a.split("_")[1]))
    if v:
        frames = T.apply_attack(v, audio, frames, np.random.default_rng(1))[1]
    if q:
        still = I.jpeg(still, q)
    return audio, frames, still


@router.get("/options3")
def options3():
    return {"plans": [{"name": k, "description": v} for k, v in PLAN_INFO.items()],
            "scenarios": list(SCENARIOS), "max_secret_bytes": MAX_SECRET_BYTES}


@router.post("/embed3")
def embed3(req: Embed3Request):
    if req.plan not in PLANS:
        raise HTTPException(400, f"plan must be one of {list(PLANS)}")
    secret = req.secret.encode()
    if len(secret) > MAX_SECRET_BYTES:
        raise HTTPException(400, f"secret is {len(secret)} bytes; the limit is {MAX_SECRET_BYTES}")
    audio, frames = load_clip(req.clip_id)
    raw = ROOT / "data" / "raw" / f"{req.clip_id}.flv"
    if not raw.exists():
        raise HTTPException(400, "the original clip is needed for the image carrier "
                                 f"(missing {raw.name} in data/raw)")
    still = I.load_still(str(raw))
    plan = PLANS[req.plan]
    risk = None
    try:
        if plan.risk:
            risk = risk_split(audio, frames, still, detectors(), plan.video)
        sa, sf, si, meta = embed(audio, frames, still, secret, derive_key(req.passphrase), plan,
                                 shares=risk["shares"] if risk else None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"cannot embed: {e}")
    job = uuid.uuid4().hex[:12]
    folder = OUT / job
    folder.mkdir()
    write_wav(folder / "stego.wav", sa)
    write_luma(str(folder / "stego.mkv"), sf)
    cv2.imwrite(str(folder / "stego.png"), si)
    (folder / "meta3.json").write_text(json.dumps({"clip_id": req.clip_id, "meta": meta}))
    used = meta["carriers"]
    each = meta.get("frag_bits_each")
    extra = {"bits_by_carrier": {c: (each[i] if each else meta["frag_bits"]) for i, c in enumerate(used)}}
    extra["shares"] = {c: round(v / sum(extra["bits_by_carrier"].values()), 3)
                       for c, v in extra["bits_by_carrier"].items()}
    extra["risk_driven"] = bool(risk)
    if risk:
        extra = {"shares": {c: round(v, 3) for c, v in risk["shares"].items()},
                 "bytes_by_carrier": dict(zip(used, meta["frag_lens"])),
                 "bits_by_carrier": dict(zip(used, each)),
                 "safe_capacity_bits": {c: round(v) for c, v in risk["safe_bits"].items()},
                 "risk_at_128_bits": {c: (None if v is None else round(v, 3))
                                      for c, v in risk["risk_at_128"].items()},
                 "risk_note": None if risk["image_detector"] else
                 "image detector not trained (python -m scripts.train_image_detector): image risk not used"}
    out = {
        "job_id": job, "plan": req.plan, "k": meta["k"], "n": meta["n"],
        "carriers": used, "bits_per_carrier": meta["frag_bits"],
        "bits_total": sum(each) if each else meta["frag_bits"] * meta["n"], **extra,
        "quality": {
            "snr_db": clean(snr(audio, sa)) if "audio" in used else None,
            "video_psnr_db": clean(psnr(frames, sf)) if "video" in used else None,
            "video_ssim": clean(ssim(frames, sf)) if "video" in used else None,
            "image_psnr_db": clean(psnr(still[None], si[None])) if "image" in used else None},
    }
    db.log_job(out, req.clip_id)
    return out


@router.post("/extract3")
def extract3(req: Extract3Request):
    folder = OUT / req.job_id
    if not (folder / "meta3.json").exists():
        raise HTTPException(404, "unknown job_id (or the job was made with /embed)")
    if req.scenario not in SCENARIOS:
        raise HTTPException(400, f"scenario must be one of {list(SCENARIOS)}")
    meta = json.loads((folder / "meta3.json").read_text())["meta"]
    audio = read_wav(folder / "stego.wav")
    frames = read_luma(str(folder / "stego.mkv"))
    still = cv2.imread(str(folder / "stego.png"), cv2.IMREAD_GRAYSCALE)
    audio, frames, still = through_scenario(req.scenario, audio, frames, still)
    secret, status = extract(audio, frames, still, derive_key(req.passphrase), meta)
    res = {"recovered": secret is not None, "scenario": req.scenario,
            "carrier_status": dict(zip(meta["carriers"], status)),
            "needed": meta["k"], "of": meta["n"],
            "secret": secret.decode(errors="replace") if secret is not None else None}
    db.log_extract(req.job_id, req.scenario, res["recovered"], res["carrier_status"])
    return res


@router.get("/history3")
def history3(limit: int = 50):
    return _history(limit)


@router.get("/download3/{job_id}/{kind}")
def download3(job_id: str, kind: str):
    names = {"audio": "stego.wav", "video": "stego.mkv", "image": "stego.png"}
    if kind not in names or not (OUT / job_id / names[kind]).exists():
        raise HTTPException(404, "not found")
    return FileResponse(OUT / job_id / names[kind], filename=f"{job_id}_{names[kind]}")
