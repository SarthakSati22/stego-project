"""DCT-domain video embedder (luma only, blind extraction, key-driven positions).

Every frame is split into 8x8 blocks and each block gets a 2-D DCT. A hidden bit
is stored in one mid-frequency coefficient using quantisation index modulation
(QIM): the coefficient is moved to the nearest multiple of DELTA whose index
parity equals the bit. Each bit can be repeated `reps` times in different,
key-chosen places; the extractor adds up the evidence from all repeats, which
makes the hidden data survive video compression much better.
"""
import numpy as np
from scipy.fft import dctn, idctn

BLOCK = 8
COEFFS = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0))   # low frequencies survive compression best
DELTA = 64.0                                 # QIM step (bigger = more robust, more visible)
REPS = 16                                    # how many times each bit is repeated


def _blocks(frames: np.ndarray) -> np.ndarray:
    t, h, w = frames.shape
    b = frames.reshape(t, h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    return b.transpose(0, 1, 3, 2, 4)        # (t, bh, bw, 8, 8)


def _unblocks(blocks: np.ndarray) -> np.ndarray:
    t, bh, bw, _, _ = blocks.shape
    return blocks.transpose(0, 1, 3, 2, 4).reshape(t, bh * BLOCK, bw * BLOCK)


def capacity_bits(shape, reps: int = REPS, coeffs=COEFFS) -> int:
    """Number of distinct bits that fit when each is repeated `reps` times."""
    t, h, w = shape
    return (t * (h // BLOCK) * (w // BLOCK) * len(coeffs)) // reps


def _slots(shape, nbits: int, reps: int, stego_key: int, coeffs=COEFFS):
    """Key-driven (frame, block row, block col, coefficient) slots, shape (reps, nbits)."""
    t, h, w = shape
    bh, bw, nc = h // BLOCK, w // BLOCK, len(coeffs)
    total = t * bh * bw * nc
    if nbits * reps > total:
        raise ValueError(f"payload of {nbits} bits x {reps} reps exceeds capacity {total}")
    order = np.random.default_rng(stego_key).permutation(total)[: nbits * reps]
    order = order.reshape(reps, nbits)
    c = order % nc
    rest = order // nc
    col = rest % bw
    rest //= bw
    row = rest % bh
    fr = rest // bh
    return fr, row, col, c


def embed(frames: np.ndarray, bits, stego_key: int, delta=DELTA, reps=REPS,
          coeffs=COEFFS) -> np.ndarray:
    """Hide bits in a (frames, H, W) uint8 luma array; returns a uint8 array."""
    bits = np.asarray(bits, dtype=np.int64)
    t, h, w = frames.shape
    dct = dctn(_blocks(frames.astype(np.float64)), axes=(3, 4), norm="ortho")
    fr, row, col, c = _slots(frames.shape, len(bits), reps, stego_key, coeffs)
    cu = np.array([k[0] for k in coeffs])[c]
    cv = np.array([k[1] for k in coeffs])[c]
    val = dct[fr, row, col, cu, cv] / delta
    q = np.rint(val)
    wrong = (q.astype(np.int64) % 2) != bits[None, :]
    q = np.where(wrong, q + np.where(val >= q, 1, -1), q)
    dct[fr, row, col, cu, cv] = q * delta
    out = _unblocks(idctn(dct, axes=(3, 4), norm="ortho"))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def extract(frames: np.ndarray, nbits: int, stego_key: int, delta=DELTA, reps=REPS,
            coeffs=COEFFS) -> np.ndarray:
    """Read nbits back from a (possibly attacked) luma array. Needs no cover."""
    dct = dctn(_blocks(frames.astype(np.float64)), axes=(3, 4), norm="ortho")
    fr, row, col, c = _slots(frames.shape, nbits, reps, stego_key, coeffs)
    cu = np.array([k[0] for k in coeffs])[c]
    cv = np.array([k[1] for k in coeffs])[c]
    u = dct[fr, row, col, cu, cv] / delta
    frac = np.mod(u, 2.0)
    d_even = np.minimum(frac, 2.0 - frac)     # distance to nearest even lattice point
    d_odd = np.abs(frac - 1.0)                # distance to nearest odd lattice point
    soft = (d_even - d_odd).sum(axis=0)       # > 0 means "odd" (bit 1)
    return (soft > 0).astype(np.uint8)
