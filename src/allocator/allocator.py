"""Payload allocator: how much of the secret goes into audio and how much into video."""
from dataclasses import dataclass

import numpy as np

from src.allocator.profiles import MEASURED, PROFILES
from src.audio import stft_embed as A
from src.pipeline import embed_clip
from src.video import dct_embed as V

STRATEGIES = ("audio_only", "video_only", "fixed_50_50",
              "adaptive_no_feedback", "adaptive_with_feedback")
WEIGHTS = dict(capacity=0.5, quality=1.0, robustness=1.0, risk=2.0)
FIXED_SHARES = {"audio_only": 1.0, "video_only": 0.0, "fixed_50_50": 0.5}


@dataclass
class Carrier:
    capacity: int       # bits it can hold at the chosen embedding strength
    quality: float      # 0..1, higher = less visible damage
    robustness: float   # 0..1, share of attacks the hidden data survives


def carriers_for(audio, frames, profile="robust"):
    """Build the audio and video Carrier descriptions for one clip."""
    p, m = PROFILES[profile], MEASURED[profile]
    a = Carrier(A.capacity_bits(len(audio), p["audio"]["reps"]),
                min(1.0, m["audio_snr"] / 40.0), m["audio_rob"])
    v = Carrier(V.capacity_bits(frames.shape, p["video"]["reps"]),
                min(1.0, m["video_psnr"] / 60.0), m["video_rob"])
    return a, v


def make_embed_fn(audio, frames, bits, profile="robust", audio_key=7, video_key=5):
    """Return embed_fn(share) -> (stego_audio, stego_frames, n_audio, n_video)."""
    p = PROFILES[profile]

    def embed_fn(share):
        return embed_clip(audio, frames, bits, share, audio_key, video_key,
                          audio_kw=p["audio"], video_kw=p["video"])
    return embed_fn


def _logit(p):
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return float(np.log(p / (1 - p)))


class Allocator:
    def __init__(self, audio: Carrier, video: Carrier, weights=None,
                 tau=0.5, rounds=4, step=0.05):
        self.audio, self.video = audio, video
        self.w = {**WEIGHTS, **(weights or {})}
        self.tau, self.rounds, self.step = tau, rounds, step

    def _score(self, c: Carrier, n_bits: int, risk: float = 0.0) -> float:
        w = self.w
        return (w["capacity"] * min(1.0, c.capacity / n_bits) + w["quality"] * c.quality
                + w["robustness"] * c.robustness - w["risk"] * risk)

    def initial_share(self, n_bits: int, risk=(0.0, 0.0)) -> float:
        """Audio share from the score gap between the two carriers."""
        d = self._score(self.audio, n_bits, risk[0]) - self._score(self.video, n_bits, risk[1])
        return float(1.0 / (1.0 + np.exp(-d / self.tau)))

    def feasible(self, share: float, n_bits: int) -> float:
        """Clamp the share so neither carrier is asked to hold more than its capacity."""
        n_a = min(int(round(share * n_bits)), self.audio.capacity)
        n_v = n_bits - n_a
        if n_v > self.video.capacity:
            n_v = self.video.capacity
            n_a = n_bits - n_v
        if n_a > self.audio.capacity or n_a < 0:
            raise ValueError("payload does not fit in the two carriers")
        return n_a / n_bits

    def run(self, strategy: str, n_bits: int, embed_fn, detect_fn=None) -> dict:
        """Choose a share, embed, and return the result.

        embed_fn(share) -> (stego_audio, stego_frames, n_audio, n_video)
        detect_fn(stego_audio, stego_frames) -> (p_audio, p_video, p_fused)
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy: {strategy}")
        if strategy in FIXED_SHARES or strategy == "adaptive_no_feedback":
            share = FIXED_SHARES.get(strategy, None)
            if share is None:
                share = self.initial_share(n_bits)
            share = self.feasible(share, n_bits)
            sa, sv, n_a, n_v = embed_fn(share)
            return dict(strategy=strategy, share=share, sa=sa, sv=sv,
                        n_a=n_a, n_v=n_v, risk=None, round=1)
        if detect_fn is None:
            raise ValueError("adaptive_with_feedback needs a detect_fn")
        share, best, best_key = self.initial_share(n_bits), None, None
        for r in range(self.rounds):
            share = self.feasible(share, n_bits)
            sa, sv, n_a, n_v = embed_fn(share)
            p_a, p_v, p = detect_fn(sa, sv)
            key = (round(float(p), 4), float(p_a + p_v))
            if best is None or key < best_key:
                best_key = key
                best = dict(strategy=strategy, share=share, sa=sa, sv=sv,
                            n_a=n_a, n_v=n_v, risk=float(p), round=r + 1)
            # move payload away from the carrier the detector finds more suspicious
            share = float(np.clip(share - self.step * (_logit(p_a) - _logit(p_v)), 0.0, 1.0))
        return best