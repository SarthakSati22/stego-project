"""Read and write the clips used by the project (luma-only video, 16-bit audio)."""
import subprocess

import numpy as np
import soundfile as sf


def _probe_size(path: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def read_luma(path: str) -> np.ndarray:
    """Decode a video file into a (frames, height, width) uint8 luma array."""
    w, h = _probe_size(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w).copy()


def write_luma(path: str, frames: np.ndarray, fps: int = 25) -> None:
    """Save a (frames, height, width) uint8 array as a lossless FFV1 video."""
    _, h, w = frames.shape
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "ffv1", str(path)],
        input=np.ascontiguousarray(frames, dtype=np.uint8).tobytes(), check=True)


def read_wav(path: str) -> np.ndarray:
    """Read a mono 16-bit WAV file as an int16 array."""
    audio, _ = sf.read(str(path), dtype="int16")
    return audio


def write_wav(path: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
