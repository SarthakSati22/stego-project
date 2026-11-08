"""k-of-n erasure coding across carriers.

The encrypted frame (nonce + ciphertext + tag) is cut into k equal fragments and
m = n - k parity fragments are computed with Reed-Solomon across the fragments
(byte column by byte column). Fragment i goes to carrier i. Any k of the n
fragments are enough to rebuild the whole frame, so losing a complete carrier
(for example the audio track being destroyed by heavy MP3 compression) no longer
loses the secret.

Each fragment is also protected by its own CRC32 and a small inner Reed-Solomon
code. If a fragment still fails its CRC after the inner repair, the receiver
treats it as *erased* (position known), which the outer code handles for free.
"""
import zlib

import numpy as np
from reedsolo import ReedSolomonError, RSCodec

from src.crypto.secret import NONCE_LEN, TAG_LEN

INNER_NSYM = 32


def frame_len(secret_len: int) -> int:
    return NONCE_LEN + secret_len + TAG_LEN


def frag_len(flen: int, k: int) -> int:
    return -(-flen // k)                                  # ceil


def fragment_nbits(flen: int, k: int, inner: int = INNER_NSYM) -> int:
    """Bits one carrier must hold."""
    return (frag_len(flen, k) + 4 + inner) * 8


def split_parity(frame: bytes, k: int, n: int) -> list[bytes]:
    """frame -> n fragments (k data + n-k parity), all of equal length."""
    L = frag_len(len(frame), k)
    padded = frame + bytes(L * k - len(frame))
    data = [padded[i * L:(i + 1) * L] for i in range(k)]
    m = n - k
    if m == 0:
        return data
    rs = RSCodec(m)
    parity = [bytearray(L) for _ in range(m)]
    for c in range(L):
        col = bytes(d[c] for d in data)
        cw = rs.encode(col)
        for j in range(m):
            parity[j][c] = cw[k + j]
    return data + [bytes(p) for p in parity]


def join_parity(frags: list, flen: int, k: int, n: int) -> bytes | None:
    """frags: list of n items (bytes, or None when erased). Returns the frame or None."""
    L = frag_len(flen, k)
    m = n - k
    erased = [i for i, f in enumerate(frags) if f is None]
    if len(erased) > m:
        return None
    if not erased:
        return b"".join(frags[:k])[:flen]
    rs = RSCodec(m)
    out = [bytearray(L) for _ in range(k)]
    for c in range(L):
        cw = bytearray(frags[i][c] if frags[i] is not None else 0 for i in range(n))
        try:
            dec = rs.decode(bytes(cw), erase_pos=erased)[0]
        except ReedSolomonError:
            return None
        for i in range(k):
            out[i][c] = dec[i]
    return b"".join(bytes(o) for o in out)[:flen]


def protect(fragment: bytes, inner: int = INNER_NSYM) -> np.ndarray:
    """CRC32 + inner Reed-Solomon, returned as bits."""
    body = fragment + zlib.crc32(fragment).to_bytes(4, "big")
    return np.unpackbits(np.frombuffer(bytes(RSCodec(inner).encode(body)), dtype=np.uint8))


def unprotect(bits, inner: int = INNER_NSYM) -> bytes | None:
    """Repair a fragment; None means 'treat as erased'."""
    raw = np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()
    try:
        body = bytes(RSCodec(inner).decode(raw)[0])
    except ReedSolomonError:
        return None
    frag, crc = body[:-4], body[-4:]
    return frag if zlib.crc32(frag).to_bytes(4, "big") == crc else None


# ---- unequal ("risk-weighted") split: n pieces whose sizes follow the given shares, no parity ----

def weighted_lengths(flen: int, shares) -> list[int]:
    """Piece sizes in bytes (each at least 1) that follow the shares and add up to flen."""
    sh = np.asarray(shares, dtype=float)
    sh = sh / sh.sum()
    lens = np.maximum(1, np.floor(sh * flen).astype(int))
    lens[int(np.argmax(lens))] += flen - int(lens.sum())
    return [int(x) for x in lens]


def split_weighted(frame: bytes, shares) -> list[bytes]:
    out, pos = [], 0
    for n in weighted_lengths(len(frame), shares):
        out.append(frame[pos:pos + n])
        pos += n
    return out


def join_weighted(frags: list, flen: int) -> bytes | None:
    """All pieces are needed: a missing one (None) loses the frame."""
    if any(f is None for f in frags):
        return None
    frame = b"".join(frags)
    return frame if len(frame) == flen else None


def piece_nbits(length: int, inner: int = INNER_NSYM) -> int:
    """Bits one carrier must hold for a piece of this many bytes."""
    return (length + 4 + inner) * 8
