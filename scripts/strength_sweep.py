"""Sweep embedding strength: quality vs. detectability vs. robustness."""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.audio import stft_embed as A
from src.channel.attacks import apply_attack
from src.crypto.secret import decode_secret, derive_key, encode_secret
from src.eval.metrics import psnr, snr
from src.media_io import read_luma, read_wav
from src.pipeline import embed_clip
from src.steganalysis.detectors import Detector, auc
from src.steganalysis.features import audio_features, video_features
from src.video import dct_embed as V

PROFILES = {
    "stealth": dict(audio=dict(delta=0.3, reps=1), video=dict(delta=12.0, reps=2)),
    "low": dict(audio=dict(delta=0.5, reps=1), video=dict(delta=24.0, reps=4)),
    "medium": dict(audio=dict(delta=0.8, reps=1), video=dict(delta=40.0, reps=8)),
    "robust": dict(audio=dict(delta=0.8, reps=2), video=dict(delta=64.0, reps=16)),
}
KEY = derive_key("sweep")
PAYLOADS = (32, 64, 128)
SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
ATTACKS = ("mp3_128", "mp3_64", "h264_crf23", "h264_crf28", "h265_crf28", "resize_75")


def features_for_clip(args):
    i, row, prof = args
    rng = np.random.default_rng(1000 + i)
    audio, frames = read_wav(row["audio_path"]), read_luma(row["video_path"])
    out = [(row["split"], 0, 0, audio_features(audio), video_features(frames))]
    for _ in range(2):
        payload = int(rng.choice(PAYLOADS))
        share = float(rng.choice(SHARES))
        bits = encode_secret(rng.bytes(payload), KEY, nonce=rng.bytes(12))
        sa, sv, n_a, n_v = embed_clip(audio, frames, bits, share,
                                      audio_key=int(rng.integers(1, 10**6)),
                                      video_key=int(rng.integers(1, 10**6)),
                                      audio_kw=prof["audio"], video_kw=prof["video"])
        out.append((row["split"], n_a, n_v, audio_features(sa), video_features(sv)))
    return out


def robustness(clips, prof):
    secret = bytes(range(64))
    bits = encode_secret(secret, KEY, nonce=bytes(12))
    n = len(bits)
    rec = {a: [] for a in ATTACKS}
    q_snr, q_psnr = [], []
    for _, r in clips.iterrows():
        audio, frames = read_wav(r["audio_path"]), read_luma(r["video_path"])
        sa = A.embed(audio, bits, 7, **prof["audio"])
        sv = V.embed(frames, bits, 5, **prof["video"])
        q_snr.append(snr(audio, sa))
        q_psnr.append(psnr(frames, sv))
        for name in ATTACKS:
            aa, vv = apply_attack(name, sa, sv, np.random.default_rng(0))
            if name.startswith("mp3"):
                got = decode_secret(A.extract(aa, n, 7, **prof["audio"]), KEY)
            else:
                got = decode_secret(V.extract(vv, n, 5, **prof["video"]), KEY)
            rec[name].append(got == secret)
    return float(np.mean(q_snr)), float(np.mean(q_psnr)), {k: float(np.mean(v)) for k, v in rec.items()}


def main() -> None:
    n_clips = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_rob = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    manifest = pd.read_csv("data/manifest.csv")
    sub = manifest.sample(min(n_clips, len(manifest)), random_state=0)
    rob_clips = sub[sub["split"] == "test"].iloc[:n_rob]
    rows = []
    with ProcessPoolExecutor() as ex:
        for name, prof in PROFILES.items():
            print(f"profile {name} ...", flush=True)
            jobs = [(i, r, prof) for i, r in enumerate(sub.to_dict("records"))]
            samples = [s for chunk in ex.map(features_for_clip, jobs, chunksize=8)
                       for s in chunk]
            split = np.array([s[0] for s in samples])
            n_a = np.array([s[1] for s in samples])
            n_v = np.array([s[2] for s in samples])
            fa = np.stack([s[3] for s in samples])
            fv = np.stack([s[4] for s in samples])
            tr, te = split == "train", split == "test"
            det_a = Detector(0).fit(fa[tr], (n_a[tr] > 0).astype(int))
            det_v = Detector(0).fit(fv[tr], (n_v[tr] > 0).astype(int))
            q_snr, q_psnr, rec = robustness(rob_clips, prof)
            rows.append(dict(profile=name, audio_snr=q_snr, video_psnr=q_psnr,
                             audio_auc=auc((n_a[te] > 0).astype(int), det_a.score(fa[te])),
                             video_auc=auc((n_v[te] > 0).astype(int), det_v.score(fv[te])),
                             **{f"rec_{k}": v for k, v in rec.items()}))
    df = pd.DataFrame(rows)
    df.to_csv("results/strength_sweep.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()