import numpy as np

from src.crypto.secret import (coded_nbits, decode_secret, derive_key,
                               encode_secret)

KEY = derive_key("test-passphrase")
NONCE = bytes(range(12))


def _flip(bits, byte_positions):
    bits = bits.copy()
    for p in byte_positions:
        bits[p * 8] ^= 1
    return bits


def test_round_trip_all_sizes():
    for n in (32, 64, 128):
        secret = bytes(range(n))
        bits = encode_secret(secret, KEY, nonce=NONCE)
        assert len(bits) == coded_nbits(n)
        assert decode_secret(bits, KEY) == secret


def test_corrects_byte_errors():
    secret = b"A" * 64
    bits = encode_secret(secret, KEY, nonce=NONCE)
    rng = np.random.default_rng(0)
    pos = rng.choice(len(bits) // 8, size=10, replace=False)
    assert decode_secret(_flip(bits, pos), KEY) == secret


def test_too_many_errors_fails():
    secret = b"A" * 64
    bits = encode_secret(secret, KEY, nonce=NONCE)
    n_bytes = len(bits) // 8
    rng = np.random.default_rng(0)
    pos = rng.choice(n_bytes, size=n_bytes // 2, replace=False)
    assert decode_secret(_flip(bits, pos), KEY) is None


def test_wrong_key_fails():
    bits = encode_secret(b"A" * 32, KEY, nonce=NONCE)
    assert decode_secret(bits, derive_key("another-passphrase")) is None