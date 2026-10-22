"""Detection function for the allocator's feedback loop (uses the trained detectors)."""
import pickle

import numpy as np

from src.steganalysis.features import audio_features, video_features


def load_detect_fn(path: str = "results/detectors.pkl"):
    """Return detect(audio, frames) -> (p_audio, p_video, p_fused)."""
    with open(path, "rb") as f:
        d = pickle.load(f)

    def detect(audio, frames):
        p_a = float(d["audio"].score(audio_features(audio)[None])[0])
        p_v = float(d["video"].score(video_features(frames)[None])[0])
        p = float(d["fusion"].score(np.array([p_a]), np.array([p_v]))[0])
        return p_a, p_v, p
    return detect