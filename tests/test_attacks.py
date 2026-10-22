import numpy as np
import pytest

from src.channel.attacks import ATTACKS, apply_attack, frame_drop, resize


def _inputs():
    rng = np.random.default_rng(0)
    audio = (rng.normal(0, 2000, 16000)).astype(np.int16)
    frames = rng.integers(0, 255, size=(20, 64, 64), dtype=np.uint8)
    return audio, frames


@pytest.mark.parametrize("name", ATTACKS)
def test_every_attack_keeps_shapes(name):
    audio, frames = _inputs()
    a, f = apply_attack(name, audio, frames, np.random.default_rng(1))
    assert a.shape == audio.shape and a.dtype == np.int16
    assert f.shape == frames.shape and f.dtype == np.uint8


def test_none_changes_nothing():
    audio, frames = _inputs()
    a, f = apply_attack("none", audio, frames, np.random.default_rng(1))
    assert (a == audio).all() and (f == frames).all()


def test_audio_attacks_leave_video_alone_and_vice_versa():
    audio, frames = _inputs()
    rng = np.random.default_rng(1)
    _, f = apply_attack("mp3_128", audio, frames, rng)
    assert (f == frames).all()
    a, _ = apply_attack("h264_crf28", audio, frames, rng)
    assert (a == audio).all()


def test_frame_drop_repeats_previous_frames():
    _, frames = _inputs()
    out = frame_drop(frames, 0.25, np.random.default_rng(0))
    changed = [i for i in range(len(frames)) if not (out[i] == frames[i]).all()]
    assert len(changed) == 5 and 0 not in changed
    for i in changed:
        assert (out[i] == out[i - 1]).all()


def test_resize_blurs_but_keeps_size():
    _, frames = _inputs()
    out = resize(frames, 0.75)
    assert out.shape == frames.shape and not (out == frames).all()


def test_unknown_attack_raises():
    audio, frames = _inputs()
    with pytest.raises(ValueError):
        apply_attack("sepia_9", audio, frames, np.random.default_rng(0))
