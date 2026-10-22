from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.channel.attacks import h264
from src.crypto.secret import decode_secret, derive_key, encode_secret
from src.eval.metrics import psnr
from src.media_io import read_luma
from src.video.dct_embed import capacity_bits, embed, extract

KEY = derive_key("test-passphrase")
NONCE = bytes(12)


def make_video(seed=0, t=30, h=128, w=128):
    """Textured test video: smooth gradients, a moving square and some noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    frames = []
    for i in range(t):
        f = 90 + 60 * np.sin(x / 17.0 + i / 9.0) + 40 * np.cos(y / 23.0)
        f[20 + i:50 + i, 30:60] += 40
        f += rng.normal(0, 6, size=(h, w))
        frames.append(np.clip(f, 0, 255))
    return np.rint(frames).astype(np.uint8)


def test_clean_round_trip_all_sizes():
    video = make_video()
    for n in (32, 64, 128):
        secret = bytes(range(n))
        bits = encode_secret(secret, KEY, nonce=NONCE)
        stego = embed(video, bits, stego_key=5)
        assert stego.dtype == np.uint8 and stego.shape == video.shape
        assert (extract(stego, len(bits), 5) == bits).all()
        assert decode_secret(extract(stego, len(bits), 5), KEY) == secret


def test_video_quality_is_acceptable():
    video = make_video()
    bits = encode_secret(bytes(range(64)), KEY, nonce=NONCE)
    assert psnr(video, embed(video, bits, stego_key=5)) > 28


def test_wrong_stego_key_gives_random_bits():
    video = make_video()
    bits = encode_secret(bytes(range(64)), KEY, nonce=NONCE)
    stego = embed(video, bits, stego_key=5)
    assert (extract(stego, len(bits), stego_key=6) != bits).mean() > 0.3


def test_too_large_payload_is_rejected():
    video = make_video()
    too_many = np.zeros(capacity_bits(video.shape) + 1, dtype=np.uint8)
    with pytest.raises(ValueError):
        embed(video, too_many, 5)


def test_survives_h264_crf23():
    video = make_video(seed=2)
    secret = bytes(range(64))
    bits = encode_secret(secret, KEY, nonce=NONCE)
    stego = embed(video, bits, stego_key=5)
    attacked = h264(stego, 23)
    assert decode_secret(extract(attacked, len(bits), 5), KEY) == secret


def test_real_clip_round_trip():
    manifest = Path("data/manifest.csv")
    if not manifest.exists():
        pytest.skip("dataset not available")
    row = pd.read_csv(manifest).iloc[0]
    video = read_luma(row.video_path)
    secret = bytes(range(128))
    bits = encode_secret(secret, KEY, nonce=NONCE)
    stego = embed(video, bits, stego_key=5)
    assert decode_secret(extract(stego, len(bits), 5), KEY) == secret
