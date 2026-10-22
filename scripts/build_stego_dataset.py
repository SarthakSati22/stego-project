"""Compute detector features for clean and stego versions of every clip."""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.crypto.secret import derive_key, encode_secret
from src.media_io import read_luma, read_wav
from src.pipeline import embed_clip
from src.steganalysis.features import audio_features, video_features

KEY = derive_key("dataset-key")
PAYLOADS = (32, 64, 128)
SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
STEGO_PER_CLIP = 2


def one_clip(args):
    i, row = args
    rng = np.random.default_rng(1000 + i)
    audio, frames = read_wav(row["audio_path"]), read_luma(row["video_path"])
    out = [dict(clip_id=row["clip_id"], split=row["split"], n_a=0, n_v=0, payload=0,
                fa=audio_features(audio), fv=video_features(frames))]
    for _ in range(STEGO_PER_CLIP):
        payload = int(rng.choice(PAYLOADS))
        share = float(rng.choice(SHARES))
        secret = rng.bytes(payload)
        bits = encode_secret(secret, KEY, nonce=rng.bytes(12))
        sa, sv, n_a, n_v = embed_clip(audio, frames, bits, share,
                                      audio_key=int(rng.integers(1, 10**6)),
                                      video_key=int(rng.integers(1, 10**6)))
        out.append(dict(clip_id=row["clip_id"], split=row["split"], n_a=n_a, n_v=n_v,
                        payload=payload, fa=audio_features(sa), fv=video_features(sv)))
    return out


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    manifest = pd.read_csv("data/manifest.csv")
    if limit:
        manifest = manifest.sample(limit, random_state=0)
    jobs = list(enumerate(manifest.to_dict("records")))
    with ProcessPoolExecutor() as ex:
        rows = [r for chunk in ex.map(one_clip, jobs, chunksize=8) for r in chunk]
    np.savez_compressed(
        "results/stego_features.npz",
        clip_id=[r["clip_id"] for r in rows], split=[r["split"] for r in rows],
        n_a=[r["n_a"] for r in rows], n_v=[r["n_v"] for r in rows],
        payload=[r["payload"] for r in rows],
        fa=np.stack([r["fa"] for r in rows]), fv=np.stack([r["fv"] for r in rows]))
    print(f"saved {len(rows)} samples from {len(manifest)} clips")


if __name__ == "__main__":
    main()
