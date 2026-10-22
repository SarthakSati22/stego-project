"""Real-codec channel attacks, applied with FFmpeg (no differentiable simulation).

Audio attacks work on int16 mono arrays, video attacks on (frames, H, W) uint8
luma arrays. Every attack returns an array of the same shape as its input.
"""
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

ATTACKS = ("none", "mp3_128", "mp3_64", "h264_crf23", "h264_crf28",
           "h265_crf23", "h265_crf28", "drop_5", "drop_10", "resize_75")


def _ffmpeg(*args: str, **kw):
    return subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, **kw)


def _fit(arr: np.ndarray, length: int) -> np.ndarray:
    """Cut or zero-pad to the original length (codecs can add or drop a few samples)."""
    out = np.zeros((length,) + arr.shape[1:], dtype=arr.dtype)
    m = min(length, len(arr))
    out[:m] = arr[:m]
    if m < length and m > 0:
        out[m:] = arr[m - 1]
    return out


def mp3(audio: np.ndarray, kbps: int, sample_rate: int = 16000) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        wav, enc, dec = (Path(d) / n for n in ("a.wav", "a.mp3", "b.wav"))
        sf.write(wav, audio, sample_rate, subtype="PCM_16")
        _ffmpeg("-i", str(wav), "-codec:a", "libmp3lame", "-b:a", f"{kbps}k", str(enc))
        _ffmpeg("-i", str(enc), "-ar", str(sample_rate), "-ac", "1",
                "-c:a", "pcm_s16le", str(dec))
        out, _ = sf.read(dec, dtype="int16")
    return _fit(out, len(audio))


def _video_codec(frames: np.ndarray, codec: str, crf: int, fps: int = 25) -> np.ndarray:
    t, h, w = frames.shape
    extra = ["-x265-params", "log-level=error"] if codec == "libx265" else []
    with tempfile.TemporaryDirectory() as d:
        enc = Path(d) / "v.mp4"
        _ffmpeg("-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(fps),
                "-i", "-", "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p",
                *extra, str(enc), input=np.ascontiguousarray(frames).tobytes())
        raw = _ffmpeg("-i", str(enc), "-f", "rawvideo", "-pix_fmt", "gray", "-",
                      capture_output=True).stdout
    out = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w)
    return _fit(out, t)


def h264(frames: np.ndarray, crf: int) -> np.ndarray:
    return _video_codec(frames, "libx264", crf)


def h265(frames: np.ndarray, crf: int) -> np.ndarray:
    return _video_codec(frames, "libx265", crf)


def frame_drop(frames: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly drop frames; a dropped frame is replaced by the previous kept frame."""
    out = frames.copy()
    n = len(frames)
    drop = rng.choice(np.arange(1, n), size=int(round(ratio * n)), replace=False)
    for i in sorted(drop):
        out[i] = out[i - 1]
    return out


def resize(frames: np.ndarray, scale: float) -> np.ndarray:
    """Shrink by `scale`, then enlarge back to the original size."""
    t, h, w = frames.shape
    sh, sw = int(round(h * scale)), int(round(w * scale))
    out = np.empty_like(frames)
    for i in range(t):
        small = cv2.resize(frames[i], (sw, sh), interpolation=cv2.INTER_AREA)
        out[i] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def apply_attack(name: str, audio: np.ndarray, frames: np.ndarray,
                 rng: np.random.Generator):
    """Apply a named attack; returns (audio, frames) after the channel."""
    if name == "none":
        return audio, frames
    kind, _, val = name.partition("_")
    if kind == "mp3":
        return mp3(audio, int(val)), frames
    if kind in ("h264", "h265"):
        crf = int(val.replace("crf", ""))
        return audio, (h264 if kind == "h264" else h265)(frames, crf)
    if kind == "drop":
        return audio, frame_drop(frames, int(val) / 100.0, rng)
    if kind == "resize":
        return audio, resize(frames, int(val) / 100.0)
    raise ValueError(f"unknown attack: {name}")
