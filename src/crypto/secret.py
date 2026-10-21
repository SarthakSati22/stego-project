"""Secret pipeline: AES-GCM encryption -> Reed-Solomon coding -> bits."""
import hashlib
import os

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from reedsolo import ReedSolomonError, RSCodec

NONCE_LEN = 12
TAG_LEN = 16


def derive_key(passphrase: str) -> bytes:
    """Turn a passphrase into a 32-byte AES key."""
    return hashlib.pbkdf2_hmac(
        "sha256", passphrase.encode(), b"stego-project-salt", 100_000, 32)


def encode_secret(secret: bytes, key: bytes, nsym: int = 32,
                  nonce: bytes | None = None) -> np.ndarray:
    """Encrypt, add error correction, and return the result as a bit array."""
    nonce = nonce or os.urandom(NONCE_LEN)
    frame = nonce + AESGCM(key).encrypt(nonce, secret, None)
    coded = bytes(RSCodec(nsym).encode(frame))
    return np.unpackbits(np.frombuffer(coded, dtype=np.uint8))


def coded_nbits(secret_len: int, nsym: int = 32) -> int:
    """How many bits encode_secret produces for a secret of this size."""
    frame_len = NONCE_LEN + secret_len + TAG_LEN
    return len(RSCodec(nsym).encode(bytes(frame_len))) * 8


def decode_secret(bits, key: bytes, nsym: int = 32) -> bytes | None:
    """Repair, decrypt and return the secret, or None if recovery failed."""
    coded = np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()
    try:
        frame = bytes(RSCodec(nsym).decode(coded)[0])
        return AESGCM(key).decrypt(frame[:NONCE_LEN], frame[NONCE_LEN:], None)
    except (ReedSolomonError, InvalidTag, ValueError):
        return None