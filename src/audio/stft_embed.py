"""STFT-domain audio embedder (blind extraction, key-driven positions).

The signal is cut into non-overlapping frames of N_FFT samples (an STFT with
a rectangular window and no overlap), so changing one frame never leaks into
another. Each hidden bit is stored in the magnitude of one frequency bin using
quantisation index modulation (QIM): the log-magnitude is moved to the nearest
lattice point whose index parity equals the bit.
"""
import numpy as np

N_FFT = 512
SAMPLE_RATE = 16000
BAND_HZ = (1000.0, 4000.0)   # mid band: away from DC and very high frequencies
DELTA = 0.5                  # QIM step in natural-log magnitude (bigger = more robust)
FLOOR = 100.0                # minimum magnitude of a used bin (above int16 rounding noise)


def _band_bins(n_fft=N_FFT, sr=SAMPLE_RATE, band=BAND_HZ):
    lo = int(np.ceil(band[0] * n_fft / sr))
    hi = int(np.floor(band[1] * n_fft / sr))
    return np.arange(lo, hi + 1)


def capacity_bits(n_samples: int, n_fft=N_FFT, band=BAND_HZ) -> int:
    """Maximum number of bits that fit in a signal of this length."""
    return (n_samples // n_fft) * len(_band_bins(n_fft, SAMPLE_RATE, band))


def _slots(n_samples: int, nbits: int, stego_key: int, n_fft=N_FFT, band=BAND_HZ):
    """Key-driven (frame, bin) positions for each bit."""
    bins = _band_bins(n_fft, SAMPLE_RATE, band)
    n_frames = n_samples // n_fft
    total = n_frames * len(bins)
    if nbits > total:
        raise ValueError(f"payload of {nbits} bits exceeds capacity {total}")
    order = np.random.default_rng(stego_key).permutation(total)[:nbits]
    return order // len(bins), bins[order % len(bins)]


def embed(audio: np.ndarray, bits, stego_key: int, delta=DELTA, floor=FLOOR,
          n_fft=N_FFT, band=BAND_HZ) -> np.ndarray:
    """Hide bits in an int16 mono signal; returns an int16 signal."""
    bits = np.asarray(bits, dtype=np.int64)
    x = audio.astype(np.float64)
    n_frames = len(x) // n_fft
    frames = x[: n_frames * n_fft].reshape(n_frames, n_fft)
    spec = np.fft.rfft(frames, axis=1)
    fr, bn = _slots(len(x), len(bits), stego_key, n_fft, band)
    q_min = int(np.ceil(np.log(floor) / delta))
    for f, b, bit in zip(fr, bn, bits):
        c = spec[f, b]
        mag = abs(c)
        phase = c / mag if mag > 0 else 1.0
        pos = np.log(max(mag, floor)) / delta
        q = int(np.rint(pos))
        if q < q_min:
            q = q_min
        if q % 2 != bit:
            q += 1 if pos >= q else -1
            if q < q_min:
                q += 2
        spec[f, b] = np.exp(q * delta) * phase
    out = x.copy()
    out[: n_frames * n_fft] = np.fft.irfft(spec, n=n_fft, axis=1).reshape(-1)
    return np.clip(np.rint(out), -32768, 32767).astype(np.int16)


def extract(audio: np.ndarray, nbits: int, stego_key: int, delta=DELTA,
            n_fft=N_FFT, band=BAND_HZ) -> np.ndarray:
    """Read nbits back from a (possibly attacked) int16 signal. Needs no cover."""
    x = audio.astype(np.float64)
    n_frames = len(x) // n_fft
    spec = np.fft.rfft(x[: n_frames * n_fft].reshape(n_frames, n_fft), axis=1)
    fr, bn = _slots(len(x), nbits, stego_key, n_fft, band)
    mags = np.abs(spec[fr, bn])
    q = np.rint(np.log(np.maximum(mags, 1e-9)) / delta).astype(np.int64)
    return (q % 2).astype(np.uint8)