"""Run all five strategies on one real test clip and print what each one does."""
import sys

import pandas as pd

from src.allocator.allocator import STRATEGIES, Allocator, carriers_for, make_embed_fn
from src.allocator.detect import load_detect_fn
from src.allocator.profiles import PROFILES
from src.crypto.secret import derive_key, encode_secret
from src.media_io import read_luma, read_wav
from src.pipeline import recover_secret

KEY = derive_key("try")


def main() -> None:
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    row = pd.read_csv("data/manifest.csv").query("split == 'test'").iloc[idx]
    audio, frames = read_wav(row["audio_path"]), read_luma(row["video_path"])
    secret = bytes(range(64))
    bits = encode_secret(secret, KEY, nonce=bytes(12))
    alloc = Allocator(*carriers_for(audio, frames, "robust"))
    embed_fn = make_embed_fn(audio, frames, bits, "robust")
    detect = load_detect_fn()
    prof = PROFILES["robust"]
    print(f"clip {row['clip_id']}, {len(bits)} bits")
    for name in STRATEGIES:
        out = alloc.run(name, len(bits), embed_fn, detect)
        p_a, p_v, p = detect(out["sa"], out["sv"])
        got = recover_secret(out["sa"], out["sv"], out["n_a"], out["n_v"], KEY,
                             audio_kw=prof["audio"], video_kw=prof["video"])
        print(f"{name:24s} audio_share={out['share']:.2f} n_a={out['n_a']:4d} "
              f"n_v={out['n_v']:4d} | detect audio={p_a:.3f} video={p_v:.3f} "
              f"fused={p:.3f} | recovered={got == secret}")


if __name__ == "__main__":
    main()