"""Three-carrier pipeline (audio, video, image) with k-of-n erasure coding.

n carriers hold n fragments; any k of them rebuild the secret.
  k = 1, n = 1  one carrier only            (no redundancy, single point of failure)
  k = n = 3     plain split                  (any lost carrier = lost secret)
  k = 2, n = 3  erasure coded                (one whole carrier may be lost)
  k = 1, n = 3  replication                  (two whole carriers may be lost)
"""
import os
from dataclasses import dataclass

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.audio import stft_embed as A
from src.crossmodal import erasure as E
from src.crossmodal import image_carrier as I
from src.crypto.secret import NONCE_LEN
from src.video import dct_embed as VS
from src.video import motion_embed as VM

KEYS = {"audio": 7, "video": 5, "image": 3}
ORDER = ("audio", "video", "image")


@dataclass
class Plan:
    carriers: tuple          # which carriers are used, in fragment order
    k: int                   # fragments needed
    video: str = "standard"  # "standard" or "motion"
    risk: bool = False       # unequal pieces chosen from detector risk (needs every carrier)

    @property
    def n(self):
        return len(self.carriers)


PLANS = {
    "audio_only": Plan(("audio",), 1),
    "video_only": Plan(("video",), 1),
    "image_only": Plan(("image",), 1),
    "split3": Plan(ORDER, 3),
    "erasure_2of3": Plan(ORDER, 2),
    "replicate_3": Plan(ORDER, 1),
    "erasure_2of3_motion": Plan(ORDER, 2, "motion"),
    "video_only_motion": Plan(("video",), 1, "motion"),
    "risk_split3": Plan(ORDER, 3, "standard", True),
    "risk_split3_motion": Plan(ORDER, 3, "motion", True),
}
DEFAULT_SHARES = {"audio": 1 / 3, "video": 1 / 3, "image": 1 / 3}   # used only if no shares are given


def _vid(plan):
    return VM if plan.video == "motion" else VS


def embed(audio, frames, still, secret: bytes, passphrase_key: bytes, plan: Plan, shares=None):
    """Returns (stego_audio, stego_frames, stego_still, meta).

    shares ({carrier: fraction}) is used only by the risk plans; it defaults to equal thirds.
    """
    nonce = os.urandom(NONCE_LEN)
    frame = nonce + AESGCM(passphrase_key).encrypt(nonce, secret, None)
    lens = None
    if plan.risk:
        shares = shares or DEFAULT_SHARES
        sh = [shares[c] for c in plan.carriers]
        lens = E.weighted_lengths(len(frame), sh)
        frags = E.split_weighted(frame, sh)
    else:
        frags = E.split_parity(frame, plan.k, plan.n)
    out = {"audio": audio, "video": frames, "image": still}
    for name, frag in zip(plan.carriers, frags):
        bits = E.protect(frag)
        if name == "audio":
            out[name] = A.embed(audio, bits, KEYS[name])
        elif name == "video":
            out[name] = _vid(plan).embed(frames, bits, KEYS[name])
        else:
            out[name] = I.embed(still, bits, KEYS[name])
    meta = {"flen": len(frame), "k": plan.k, "n": plan.n, "carriers": plan.carriers,
            "video": plan.video, "frag_bits": E.fragment_nbits(len(frame), plan.k)}
    if plan.risk:
        meta.update(risk=True, frag_lens=lens, frag_bits_each=[E.piece_nbits(x) for x in lens],
                    frag_bits=max(E.piece_nbits(x) for x in lens),
                    shares={c: float(x) for c, x in zip(plan.carriers, sh)})
    return out["audio"], out["video"], out["image"], meta


def extract(audio, frames, still, passphrase_key: bytes, meta: dict):
    """Returns (secret_or_None, list of per-carrier status 'ok'/'erased')."""
    plan = Plan(tuple(meta["carriers"]), meta["k"], meta["video"], bool(meta.get("risk", False)))
    nbs = meta["frag_bits_each"] if plan.risk else [meta["frag_bits"]] * plan.n
    frags, status = [], []
    for name, nb in zip(plan.carriers, nbs):
        if name == "audio":
            bits = A.extract(audio, nb, KEYS[name])
        elif name == "video":
            bits = _vid(plan).extract(frames, nb, KEYS[name])
        else:
            bits = I.extract(still, nb, KEYS[name])
        f = E.unprotect(bits)
        frags.append(f)
        status.append("ok" if f is not None else "erased")
    frame = (E.join_weighted(frags, meta["flen"]) if plan.risk
             else E.join_parity(frags, meta["flen"], plan.k, plan.n))
    if frame is None:
        return None, status
    try:
        return AESGCM(passphrase_key).decrypt(frame[:NONCE_LEN], frame[NONCE_LEN:], None), status
    except (InvalidTag, ValueError):
        return None, status
