import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio.stft_embed import capacity_bits, embed, extract
from src.crypto.secret import decode_secret, derive_key, encode_secret

KEY = derive_key("test-passphrase")
NONCE = bytes(12)


def make_audio(seed=0, n=32000, rms=2000):
    """Speech-like test signal: harmonics + slow envelope + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    f0 = 120 + 20 * np.sin(2 * np.pi * 3 * t)
    sig = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 30))
    sig = sig * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t) ** 2)
    sig = sig + 0.1 * rng.standard_normal(n)
    sig = sig / np.sqrt((sig ** 2).mean()) * rms
    return np.rint(sig).astype(np.int16)


def snr_db(a, b):
    a, b = a.astype(float), b.astype(float)
    return 10 * np.log10((a ** 2).sum() / ((a - b) ** 2).sum())


def test_clean_round_trip_all_sizes():
    audio = make_audio()
    for n in (32, 64, 128):
        secret = bytes(range(n))
        bits = encode_secret(secret, KEY, nonce=NONCE)
        stego = embed(audio, bits, stego_key=7)
        assert stego.dtype == np.int16 and len(stego) == len(audio)
        assert (extract(stego, len(bits), 7) == bits).all()
        assert decode_secret(extract(stego, len(bits), 7), KEY) == secret


def test_audio_quality_is_acceptable():
    audio = make_audio()
    bits = encode_secret(bytes(range(64)), KEY, nonce=NONCE)
    assert snr_db(audio, embed(audio, bits, stego_key=7)) > 20


def test_wrong_stego_key_gives_random_bits():
    audio = make_audio()
    bits = encode_secret(bytes(range(64)), KEY, nonce=NONCE)
    stego = embed(audio, bits, stego_key=7)
    wrong = extract(stego, len(bits), stego_key=8)
    assert (wrong != bits).mean() > 0.3


def test_too_large_payload_is_rejected():
    audio = make_audio()
    with pytest.raises(ValueError):
        embed(audio, np.zeros(capacity_bits(len(audio)) + 1, dtype=np.uint8), 7)


def test_survives_mp3_128k():
    audio = make_audio(seed=1)
    secret = bytes(range(64))
    bits = encode_secret(secret, KEY, nonce=NONCE)
    stego = embed(audio, bits, stego_key=7)
    with tempfile.TemporaryDirectory() as d:
        wav, mp3, back = (Path(d) / n for n in ("a.wav", "a.mp3", "b.wav"))
        sf.write(wav, stego, 16000, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav),
                        "-codec:a", "libmp3lame", "-b:a", "128k", str(mp3)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp3), "-ar", "16000",
                        "-ac", "1", "-c:a", "pcm_s16le", str(back)], check=True)
        attacked, _ = sf.read(back, dtype="int16")
    fixed = np.zeros(len(stego), dtype=np.int16)
    m = min(len(stego), len(attacked))
    fixed[:m] = attacked[:m]
    assert decode_secret(extract(fixed, len(bits), 7), KEY) == secret