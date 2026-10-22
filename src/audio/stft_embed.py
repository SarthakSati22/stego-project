"""STFT-domain audio embedder (blind extraction, key-driven positions).

The signal is cut into non-overlapping frames of N_FFT samples (an STFT with
a rectangular window and no overlap), so changing one frame never leaks into
another. Each hidden bit is stored in the magnitude of one frequency bin using
quantisation index modulation (QIM): the log-magnitude is moved to the nearest
lattice point whose index parity equals the bit. Each bit can be repeated `reps`
times in different, key-chosen places; the extractor adds up the evidence from
all repeats, which makes the hidden data survive MP3 compression better.
"""
import numpy as np

N_FFT = 512
SAMPLE_RATE = 16000
BAND_HZ = (1000.0, 4000.0)   # mid band: away from DC and very high frequencies
DELTA = 0.8                  # QIM step in natural-log magnitude (bigger = more robust)
FLOOR = 100.0                # minimum magnitude of a used bin (above int16 rounding noise)
REPS = 2                     # how many times each bit is repeated


def _band_bins(n_fft=N_FFT, sr=SAMPLE_RATE, band=BAND_HZ):
    lo = int(np.ceil(band[0] * n_fft / sr))
    hi = int(np.floor(band[1] * n_fft / sr))
    return np.arange(lo, hi + 1)


def capacity_bits(n_samples: int, reps: int = REPS, n_fft=N_FFT, band=BAND_HZ) -> int:
    """Number of distinct bits that fit when each is repeated `reps` times."""
    return ((n_samples // n_fft) * len(_band_bins(n_fft, SAMPLE_RATE, band))) // reps


def _slots(n_samples: int, nbits: int, reps: int, stego_key: int,
           n_fft=N_FFT, band=BAND_HZ):
    """Key-driven (frame, bin) positions, shape (reps, nbits) each."""
    bins = _band_bins(n_fft, SAMPLE_RATE, band)
    n_frames = n_samples // n_fft
    total = n_frames * len(bins)
    if nbits * reps > total:
        raise ValueError(f"payload of {nbits} bits x {reps} reps exceeds capacity {total}")
    order = np.random.default_rng(stego_key).permutation(total)[: nbits * reps]
    order = order.reshape(reps, nbits)
    return order // len(bins), bins[order % len(bins)]


def embed(audio: np.ndarray, bits, stego_key: int, delta=DELTA, floor=FLOOR,
          reps=REPS, n_fft=N_FFT, band=BAND_HZ) -> np.ndarray:
    """Hide bits in an int16 mono signal; returns an int16 signal."""
    bits = np.asarray(bits, dtype=np.int64)
    x = audio.astype(np.float64)
    n_frames = len(x) // n_fft
    frames = x[: n_frames * n_fft].reshape(n_frames, n_fft)
    spec = np.fft.rfft(frames, axis=1)
    fr, bn = _slots(len(x), len(bits), reps, stego_key, n_fft, band)
    q_min = int(np.ceil(np.log(floor) / delta))
    for f, b, bit in zip(fr.ravel(), bn.ravel(), np.tile(bits, reps)):
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
            reps=REPS, n_fft=N_FFT, band=BAND_HZ) -> np.ndarray:
    """Read nbits back from a (possibly attacked) int16 signal. Needs no cover."""
    x = audio.astype(np.float64)
    n_frames = len(x) // n_fft
    spec = np.fft.rfft(x[: n_frames * n_fft].reshape(n_frames, n_fft), axis=1)
    fr, bn = _slots(len(x), nbits, reps, stego_key, n_fft, band)
    u = np.log(np.maximum(np.abs(spec[fr, bn]), 1e-9)) / delta
    frac = np.mod(u, 2.0)
    d_even = np.minimum(frac, 2.0 - frac)     # distance to nearest even lattice point
    d_odd = np.abs(frac - 1.0)                # distance to nearest odd lattice point
    return ((d_even - d_odd).sum(axis=0) > 0).astype(np.uint8)
