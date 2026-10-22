"""Train the audio, video and fusion detectors and evaluate them (with ablation)."""
import json
import pickle

import numpy as np

from src.steganalysis.detectors import Detector, FusionDetector, auc


def load(path="results/stego_features.npz"):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def subset(d, split):
    m = d["split"] == split
    return {k: v[m] for k, v in d.items()}


def main() -> None:
    d = load()
    tr, va, te = (subset(d, s) for s in ("train", "val", "test"))
    label = lambda s: dict(a=(s["n_a"] > 0).astype(int), v=(s["n_v"] > 0).astype(int),
                           any=((s["n_a"] + s["n_v"]) > 0).astype(int))
    ltr, lva, lte = label(tr), label(va), label(te)

    det_a = Detector(0).fit(tr["fa"], ltr["a"])
    det_v = Detector(0).fit(tr["fv"], ltr["v"])
    # Fusion weights are learned on the validation split.
    fusion = FusionDetector().fit(det_a.score(va["fa"]), det_v.score(va["fv"]), lva["any"])

    sa, sv = det_a.score(te["fa"]), det_v.score(te["fv"])
    res = {
        "audio_auc": auc(lte["a"], sa),
        "video_auc": auc(lte["v"], sv),
        "fusion_auc": auc(lte["any"], fusion.score(sa, sv)),
        "fusion_audio_masked_auc": auc(lte["any"], fusion.score(sa, sv, mask_audio=True)),
        "fusion_video_masked_auc": auc(lte["any"], fusion.score(sa, sv, mask_video=True)),
        "fusion_weights_audio_video": [float(w) for w in fusion.weights],
        "n_train": int(len(tr["n_a"])), "n_val": int(len(va["n_a"])), "n_test": int(len(te["n_a"])),
    }
    # Detectability by how many bits each carrier holds
    for name, s, n in (("audio", sa, te["n_a"]), ("video", sv, te["n_v"])):
        clean = n == 0
        for lo, hi in ((1, 600), (600, 1200), (1200, 10**6)):
            m = (n >= lo) & (n < hi)
            if m.sum() > 2:
                res[f"{name}_auc_bits_{lo}_{hi}"] = auc(
                    np.r_[np.zeros(clean.sum()), np.ones(m.sum())], np.r_[s[clean], s[m]])
    print(json.dumps(res, indent=2))
    with open("results/detector_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    with open("results/detectors.pkl", "wb") as f:
        pickle.dump(dict(audio=det_a, video=det_v, fusion=fusion), f)


if __name__ == "__main__":
    main()