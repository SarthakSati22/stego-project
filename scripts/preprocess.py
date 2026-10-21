"""Convert raw CREMA-D clips to the fixed format and build the manifest."""
import csv
import subprocess
from pathlib import Path

import yaml

RAW = Path("data/raw")
OUT = Path("data/processed")


def probe(path: Path, *args: str) -> str:
    cmd = ["ffprobe", "-v", "error", *args, str(path)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def run_ffmpeg(*args: str) -> bool:
    cmd = ["ffmpeg", "-v", "error", "-y", *args]
    return subprocess.run(cmd).returncode == 0


def main() -> None:
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    secs = d["clip_seconds"]
    sr, w, h, fps = (d["audio"]["sample_rate"], d["video"]["width"],
                     d["video"]["height"], d["video"]["fps"])

    # Split by actor, using the fractions in the config
    actors = sorted({p.stem.split("_")[0] for p in RAW.glob("*.flv")})
    n_train = round(len(actors) * d["split"]["train"])
    n_val = round(len(actors) * d["split"]["val"])
    actor_split = {}
    for i, a in enumerate(actors):
        actor_split[a] = ("train" if i < n_train else
                          "val" if i < n_train + n_val else "test")

    OUT.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for src in sorted(RAW.glob("*.flv")):
        parts = src.stem.split("_")
        actor, emotion = parts[0], parts[2]
        if actor not in actor_split:
            continue
        dur = probe(src, "-show_entries", "format=duration",
                    "-of", "default=nw=1:nk=1")
        has_a = probe(src, "-select_streams", "a",
                      "-show_entries", "stream=codec_type", "-of", "csv=p=0")
        has_v = probe(src, "-select_streams", "v",
                      "-show_entries", "stream=codec_type", "-of", "csv=p=0")
        try:
            ok = float(dur) >= secs and "audio" in has_a and "video" in has_v
        except ValueError:
            ok = False
        if not ok:
            skipped += 1
            continue
        vid, aud = OUT / f"{src.stem}.mkv", OUT / f"{src.stem}.wav"
        if not (vid.exists() and aud.exists()):
            good = run_ffmpeg("-i", str(src), "-t", str(secs), "-an",
                              "-vf", f"crop=ih:ih,scale={w}:{h},fps={fps}",
                              "-c:v", "ffv1", str(vid))
            good = good and run_ffmpeg("-i", str(src), "-t", str(secs), "-vn",
                                       "-ar", str(sr), "-ac", "1",
                                       "-c:a", "pcm_s16le", str(aud))
            if not good:
                skipped += 1
                continue
        rows.append([src.stem, vid.as_posix(), aud.as_posix(),
                     actor, emotion, actor_split[actor]])

    with open(d["manifest"], "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["clip_id", "video_path", "audio_path",
                     "actor", "emotion", "split"])
        wr.writerows(rows)

    counts = {s: sum(1 for r in rows if r[5] == s)
              for s in ("train", "val", "test")}
    print(f"Kept {len(rows)} clips, skipped {skipped}. Split: {counts}")


if __name__ == "__main__":
    main()