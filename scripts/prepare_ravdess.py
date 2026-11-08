"""Convert RAVDESS speech clips (audio-visual .mp4) to the project format.

Usage: python -m scripts.prepare_ravdess <folder_with_RAVDESS_files>
"""
import csv
import subprocess
import sys
from pathlib import Path

import yaml

OUT = Path("data/processed_ravdess")


def probe(path, *args):
    cmd = ["ffprobe", "-v", "error", *args, str(path)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def run_ffmpeg(*args):
    return subprocess.run(["ffmpeg", "-v", "error", "-y", *args]).returncode == 0


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m scripts.prepare_ravdess <folder>")
    src_dir = Path(sys.argv[1])
    cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))["data"]
    secs = cfg["clip_seconds"]
    sr, w, h, fps = (cfg["audio"]["sample_rate"], cfg["video"]["width"],
                     cfg["video"]["height"], cfg["video"]["fps"])
    files = sorted(p for p in src_dir.rglob("01-01-*.mp4"))
    if not files:
        sys.exit(f"no 01-01-*.mp4 files found under {src_dir}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for src in files:
        parts = src.stem.split("-")
        actor, emotion = parts[6], parts[2]
        dur = probe(src, "-show_entries", "format=duration", "-of", "default=nw=1:nk=1")
        has_a = probe(src, "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0")
        try:
            ok = float(dur) >= secs and "audio" in has_a
        except ValueError:
            ok = False
        if not ok:
            skipped += 1
            continue
        t0 = "0.5" if float(dur) >= secs + 0.5 else "0"
        vid, aud = OUT / f"{src.stem}.mkv", OUT / f"{src.stem}.wav"
        if not (vid.exists() and aud.exists()):
            good = run_ffmpeg("-ss", t0, "-i", str(src), "-t", str(secs), "-an",
                              "-vf", f"crop=ih:ih,scale={w}:{h},fps={fps}", "-c:v", "ffv1", str(vid))
            good = good and run_ffmpeg("-ss", t0, "-i", str(src), "-t", str(secs), "-vn",
                                       "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(aud))
            if not good:
                skipped += 1
                continue
        rows.append([src.stem, vid.as_posix(), aud.as_posix(), actor, emotion, "external"])
    with open("data/manifest_ravdess.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["clip_id", "video_path", "audio_path", "actor", "emotion", "split"])
        wr.writerows(rows)
    print(f"Kept {len(rows)} clips, skipped {skipped}, actors: {len({r[3] for r in rows})}")


if __name__ == "__main__":
    main()