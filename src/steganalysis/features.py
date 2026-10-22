"""Generic statistical features for steganalysis (no knowledge of keys or step sizes)."""
import numpy as np
from scipy.fft import dctn

from src.video.dct_embed import _blocks

N_FFT = 512
BAND = slice(31, 130)                     # roughly 1-4 kHz bins
DCT_POS = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2))


def _hist(x, lo, hi, bins):
    h, _ = np.histogram(np.clip(x, lo, hi - 1e-9), bins=bins, range=(lo, hi))
    return h / max(1, h.sum())


def _moments(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    s = x.std() + 1e-9
    return [x.mean(), x.std(), np.mean(np.abs(x - x.mean())), np.mean(((x - x.mean()) / s) ** 4)]


def audio_features(audio: np.ndarray) -> np.ndarray:
    """Statistics of the log-magnitude STFT in the mid band."""
    n = len(audio) // N_FFT
    spec = np.fft.rfft(audio[: n * N_FFT].astype(np.float64).reshape(n, N_FFT), axis=1)
    L = np.log(np.maximum(np.abs(spec), 1.0))[:, BAND]
    r_f = L[:, 1:-1] - 0.5 * (L[:, :-2] + L[:, 2:])        # residual along frequency
    r_t = L[1:-1] - 0.5 * (L[:-2] + L[2:])                 # residual along time
    feats = [_hist(L, 0, 12, 24), _hist(r_f, -3, 3, 24), _hist(r_t, -3, 3, 24),
             _moments(L), _moments(r_f), _moments(r_t),
             [np.mean(L < t) for t in (3.4, 4.1, 4.6, 5.0)]]   # share of weak bins
    return np.concatenate([np.ravel(f) for f in feats])


def video_features(frames: np.ndarray) -> np.ndarray:
    """DCT-coefficient histograms, pixel residuals and blockiness of the luma frames."""
    f = frames.astype(np.float64)
    dct = dctn(_blocks(f), axes=(3, 4), norm="ortho")
    feats = [_hist(dct[..., u, v], -160, 160, 40) for u, v in DCT_POS]
    rh = f[:, :, 1:-1] - 0.5 * (f[:, :, :-2] + f[:, :, 2:])
    rv = f[:, 1:-1] - 0.5 * (f[:, :-2] + f[:, 2:])
    rt = f[1:] - f[:-1]
    feats += [_hist(np.round(r), -6, 7, 13) for r in (rh, rv, rt)]
    feats += [_moments(r) for r in (rh, rv, rt)]
    dx = np.abs(np.diff(f, axis=2))
    dy = np.abs(np.diff(f, axis=1))
    feats.append([dx[:, :, 7::8].mean() / (dx.mean() + 1e-9),
                  dy[:, 7::8, :].mean() / (dy.mean() + 1e-9)])
    return np.concatenate([np.ravel(x) for x in feats])