import shutil
import subprocess

import numpy as np

from src.utils import load_config, set_seed


def test_ffmpeg_and_encoders():
    assert shutil.which("ffmpeg") is not None, "ffmpeg not on PATH"
    out = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True).stdout
    for enc in ("libx264", "libx265", "libmp3lame"):
        assert enc in out, f"{enc} missing"


def test_seed_is_repeatable():
    set_seed(1)
    a = np.random.rand(5)
    set_seed(1)
    b = np.random.rand(5)
    assert (a == b).all()


def test_config_loads():
    cfg = load_config()
    assert cfg["data"]["audio"]["sample_rate"] == 16000
    assert len(cfg["strategies"]) == 5