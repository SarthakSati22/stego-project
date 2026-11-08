"""Test the CREMA-D-trained audio/video/fusion detectors on a NEW dataset (no retraining).

Usage: python -m scripts.cross_dataset_av [manifest=data/manifest_ravdess.csv] [name=ravdess]
"""
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.build_stego_dataset import one_clip
from scripts.detector_report import metrics


def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 else "data/manifest_ravdess.csv"
    name = sys.argv[2] if len(sys.argv) > 2 else "ravdess"
    m = pd.read_csv(manifest)
    jobs = list(enumerate(m.to_dict("records")))
    with ProcessPoolExecutor() as ex:
        rows = [r for chunk in ex.map(one_clip, jobs, chunksize=4) for r in chunk]
    fa, fv = np.stack([r["fa"] for r in rows]), np.stack([r["fv"] for r in rows])
    n_a = np.array([r["n_a"] for r in rows]); n_v = np.array([r["n_v"] for r in rows])
    with open("results/detectors.pkl", "rb") as f:
        det = pickle.load(f)
    sa, sv = det["audio"].score(fa), det["video"].score(fv)
    out = {
        "audio detector": metrics((n_a > 0).astype(int), sa),
        "video detector": metrics((n_v > 0).astype(int), sv),
        "fusion detector": metrics(((n_a + n_v) > 0).astype(int), det["fusion"].score(sa, sv)),
    }
    t = pd.DataFrame(out).T
    pd.set_option("display.width", 200)
    cols = ["samples", "accuracy", "acc_ci_low", "acc_ci_high", "precision", "recall", "f1",
            "false_positive_rate", "false_negative_rate", "auc"]
    print(f"\nDetectors trained on CREMA-D, tested on {name} (clips: {len(m)}, "
          f"actors: {m['actor'].nunique()}), threshold 0.5:\n")
    print(t[cols].astype(float).round(4).to_string())
    t.to_csv(f"results/tables/{name}_cross_dataset.csv")
    print(f"\nsaved results/tables/{name}_cross_dataset.csv")


if __name__ == "__main__":
    main()