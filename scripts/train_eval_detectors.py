"""Train an independent set of detectors (random forests) used only for evaluation."""
import json
import pickle

import numpy as np

from src.steganalysis.detectors import FusionDetector, auc
from src.steganalysis.rf_detector import RFDetector


def load(path="results/stego_features.npz"):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def subset(d, split):
    m = d["split"] == split
    return {k: v[m] for k, v in d.items()}


def labels(s):
    return ((s["n_a"] > 0).astype(int), (s["n_v"] > 0).astype(int),
            ((s["n_a"] + s["n_v"]) > 0).astype(int))


def main() -> None:
    d = load()
    tr, va, te = (subset(d, s) for s in ("train", "val", "test"))
    ya_tr, yv_tr, _ = labels(tr)
    _, _, yany_va = labels(va)
    ya_te, yv_te, yany_te = labels(te)

    det_a = RFDetector(1).fit(tr["fa"], ya_tr)
    det_v = RFDetector(1).fit(tr["fv"], yv_tr)
    fusion = FusionDetector().fit(det_a.score(va["fa"]), det_v.score(va["fv"]), yany_va)

    sa, sv = det_a.score(te["fa"]), det_v.score(te["fv"])
    res = dict(audio_auc=auc(ya_te, sa), video_auc=auc(yv_te, sv),
               fusion_auc=auc(yany_te, fusion.score(sa, sv)))
    print(json.dumps(res, indent=2))
    with open("results/eval_detectors.pkl", "wb") as f:
        pickle.dump(dict(audio=det_a, video=det_v, fusion=fusion), f)


if __name__ == "__main__":
    main()