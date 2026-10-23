"""Run the full comparison: strategies x payloads x seeds x channels, on test clips."""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from src.allocator.allocator import STRATEGIES, Allocator, carriers_for, make_embed_fn
from src.allocator.detect import load_detect_fn
from src.allocator.profiles import PROFILES
from src.channel.attacks import apply_attack
from src.crypto.secret import decode_secret, derive_key, encode_secret
from src.eval.metrics import bit_error_rate, psnr, snr, ssim
from src.media_io import read_luma, read_wav
from src.pipeline import extract_clip

KEY = derive_key("experiment")
PAYLOADS = (32, 128)
CHANNELS = ("none", "mp3_128", "mp3_64", "h264_crf23", "h264_crf28", "h265_crf23",
            "h265_crf28", "drop_5", "drop_10", "resize_75",
            "mp3_128+h264_crf23", "mp3_64+h264_crf28")


@lru_cache(maxsize=None)
def detector(path):
    return load_detect_fn(path)


def finite(x):
    return float(x) if np.isfinite(x) else float("nan")


def through_channel(name, audio, frames, seed):
    """Apply a channel; 'a+b' means attack a then attack b (audio and video together)."""
    rng = np.random.default_rng(seed)
    for part in name.split("+"):
        audio, frames = apply_attack(part, audio, frames, rng)
    return audio, frames


def run_job(job):
    i, clip, payload, seed, profile = job
    prof = PROFILES[profile]
    rng = np.random.default_rng([seed, i, payload])
    audio, frames = read_wav(clip["audio_path"]), read_luma(clip["video_path"])
    secret = rng.bytes(payload)
    bits = encode_secret(secret, KEY, nonce=rng.bytes(12))
    n = len(bits)
    audio_key, video_key = int(rng.integers(1, 10**6)), int(rng.integers(1, 10**6))

    alloc = Allocator(*carriers_for(audio, frames, profile))
    embed_fn = make_embed_fn(audio, frames, bits, profile, audio_key, video_key)
    detect = detector("results/detectors.pkl")        # used by the allocator's feedback
    judge = detector("results/eval_detectors.pkl")    # independent, used only to evaluate

    rows = []
    for strategy in STRATEGIES:
        out = alloc.run(strategy, n, embed_fn, detect)
        sa, sv, n_a, n_v = out["sa"], out["sv"], out["n_a"], out["n_v"]
        pa, pv, pf = detect(sa, sv)
        ja, jv, jf = judge(sa, sv)
        base = dict(clip_id=clip["clip_id"], payload=payload, seed=seed,
                    strategy=strategy, share=out["share"], n_a=n_a, n_v=n_v,
                    snr=finite(snr(audio, sa)), psnr=finite(psnr(frames, sv)),
                    ssim=ssim(frames, sv),
                    alloc_p_audio=pa, alloc_p_video=pv, alloc_p_fused=pf,
                    judge_p_audio=ja, judge_p_video=jv, judge_p_fused=jf)
        for ch in CHANNELS:
            aa, vv = through_channel(ch, sa, sv, seed + 17)
            got = extract_clip(aa, vv, n_a, n_v, audio_key, video_key,
                               audio_kw=prof["audio"], video_kw=prof["video"])
            ok = decode_secret(got, KEY) == secret
            rows.append({**base, "channel": ch,
                         "ber": bit_error_rate(bits, got), "recovered": bool(ok)})
    return rows


def main() -> None:
    n_clips = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    profile = sys.argv[3] if len(sys.argv) > 3 else "robust"
    manifest = pd.read_csv("data/manifest.csv")
    test = manifest[manifest["split"] == "test"]
    clips = test.sample(min(n_clips, len(test)), random_state=0).to_dict("records")
    jobs = [(i, c, p, s, profile) for i, c in enumerate(clips)
            for p in PAYLOADS for s in range(n_seeds)]
    out_path = f"results/experiments_{profile}.csv"
    workers = max(1, min(8, os.cpu_count() or 1))
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(run_job, jobs), 1):
            rows.extend(r)
            if k % 10 == 0 or k == len(jobs):
                print(f"  done {k}/{len(jobs)}", flush=True)
            if k % 40 == 0:
                pd.DataFrame(rows).to_csv(out_path, index=False)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()