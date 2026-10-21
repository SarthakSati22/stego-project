"""Download a subset of CREMA-D video clips from GitHub (Git LFS), in parallel."""
import itertools
import random
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = ("https://media.githubusercontent.com/media/"
        "CheyneyComputerScience/CREMA-D/master/VideoFlash/")
OUT = Path("data/raw")
ACTORS = range(1001, 1092)      # all 91 actors
CLIPS_PER_ACTOR = 25            # ~2,200 clips total
THREADS = 16
SENTENCES = ["IEO", "TIE", "IOM", "IWW", "TAI", "MTI",
             "IWL", "ITH", "DFA", "ITS", "TSI", "WSI"]
EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
LEVELS = ["XX", "LO", "MD", "HI"]


def fetch(name: str) -> bool:
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    try:
        with urllib.request.urlopen(BASE + name, timeout=60) as r:
            data = r.read()
    except Exception:
        return False
    if len(data) < 10_000:      # LFS pointer stub, not a real video
        return False
    dest.write_bytes(data)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    total = 0
    with ThreadPoolExecutor(THREADS) as ex:
        for actor in ACTORS:
            names = [f"{actor}_{s}_{e}_{l}.flv" for s, e, l in
                     itertools.product(SENTENCES, EMOTIONS, LEVELS)]
            rng.shuffle(names)
            got, i = 0, 0
            while got < CLIPS_PER_ACTOR and i < len(names):
                chunk = names[i:i + THREADS * 2]
                i += len(chunk)
                got += sum(ex.map(fetch, chunk))
            print(f"actor {actor}: {got} clips", flush=True)
            total += got
    if total == 0:
        sys.exit("Nothing downloaded; check the URL or your connection.")
    print("Total clips:", total)


if __name__ == "__main__":
    main()