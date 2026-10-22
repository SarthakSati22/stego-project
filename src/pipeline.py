"""Split-payload pipeline: one secret hidden across the audio and video of a clip."""
import numpy as np

from src.audio import stft_embed as A
from src.crypto.secret import decode_secret, encode_secret
from src.video import dct_embed as V

AUDIO_KEY = 7      # default stego keys (shared secrets between sender and receiver)
VIDEO_KEY = 5


def split_counts(n_bits: int, share_audio: float) -> tuple[int, int]:
    """How many of the n_bits go to audio and to video."""
    n_a = int(round(share_audio * n_bits))
    return n_a, n_bits - n_a


def embed_clip(audio, frames, bits, share_audio, audio_key=AUDIO_KEY,
               video_key=VIDEO_KEY, audio_kw=None, video_kw=None):
    """Hide `bits` across the clip. Returns (stego_audio, stego_frames, n_audio, n_video)."""
    audio_kw, video_kw = audio_kw or {}, video_kw or {}
    n_a, n_v = split_counts(len(bits), share_audio)
    sa = A.embed(audio, bits[:n_a], audio_key, **audio_kw) if n_a else audio
    sv = V.embed(frames, bits[n_a:], video_key, **video_kw) if n_v else frames
    return sa, sv, n_a, n_v


def extract_clip(audio, frames, n_a, n_v, audio_key=AUDIO_KEY, video_key=VIDEO_KEY,
                 audio_kw=None, video_kw=None) -> np.ndarray:
    """Read the bits back from both carriers and join them."""
    audio_kw, video_kw = audio_kw or {}, video_kw or {}
    parts = []
    if n_a:
        parts.append(A.extract(audio, n_a, audio_key, **audio_kw))
    if n_v:
        parts.append(V.extract(frames, n_v, video_key, **video_kw))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)


def recover_secret(audio, frames, n_a, n_v, key, **kw):
    """Extract, repair and decrypt. Returns the secret bytes, or None on failure."""
    return decode_secret(extract_clip(audio, frames, n_a, n_v, **kw), key)


__all__ = ["split_counts", "embed_clip", "extract_clip", "recover_secret", "encode_secret"]