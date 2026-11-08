"""Train and save the image detector used by the risk-weighted split.

Usage: python -m scripts.train_image_detector [n_images=300]
Needs data/images/cremad.csv (python -m scripts.prepare_images cremad).
Trains on the TRAIN actors only, hides 100, 300 and 1000 random bits in each picture,
reports AUC on the TEST actors for each size, and saves results/image_detector.pkl.
"""
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from src.crossmodal import image_carrier as I
from src.crossmodal.image_features import image_features
from src.steganalysis.detectors import Detector, auc

SIZES = (100, 300, 1000)


def one(args):
    path, split, seed = args
    rng = np.random.default_rng(seed)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    rows = [dict(split=split, n=0, f=image_features(img))]
    for nb in SIZES:
        s = I.embed(img, rng.integers(0, 2, nb), 3)       # key 3 = the image carrier's key
        rows.append(dict(split=split, n=nb, f=image_features(s)))
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    df = pd.read_csv("data/images/cremad.csv")
    parts = [df[df.split == "train"].sample(min(n, (df.split == "train").sum()), random_state=0),
             df[df.split == "test"].sample(min(n // 3, (df.split == "test").sum()), random_state=0)]
    d = pd.concat(parts)
    jobs = [(r.path, r.split, i) for i, r in enumerate(d.itertuples())]
    with ProcessPoolExecutor() as ex:
        rec = [x for chunk in ex.map(one, jobs, chunksize=4) for x in chunk]
    tr = [r for r in rec if r["split"] == "train"]
    X = np.stack([r["f"] for r in tr]); y = np.array([r["n"] > 0 for r in tr], dtype=int)
    det = Detector(0).fit(X, y)
    te = [r for r in rec if r["split"] == "test"]
    clean = [r for r in te if r["n"] == 0]
    print(f"trained on {len(tr)} samples; test pictures: {len(clean)}")
    for nb in SIZES:
        st = [r for r in te if r["n"] == nb]
        Xt = np.stack([r["f"] for r in clean + st])
        yt = np.r_[np.zeros(len(clean)), np.ones(len(st))]
        print(f"  AUC at {nb:5d} hidden bits: {auc(yt, det.score(Xt)):.4f}")
    with open("results/image_detector.pkl", "wb") as f:
        pickle.dump(det, f)
    print("saved results/image_detector.pkl")


if __name__ == "__main__":
    main()
