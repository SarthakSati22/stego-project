import numpy as np

from src.eval.metrics import bit_error_rate, psnr, snr, ssim


def test_identical_signals():
    x = np.random.default_rng(0).integers(0, 255, size=(3, 32, 32), dtype=np.uint8)
    assert psnr(x, x) == float("inf")
    assert abs(ssim(x, x) - 1.0) < 1e-9
    assert snr(x.ravel(), x.ravel()) == float("inf")


def test_psnr_known_value():
    a = np.zeros((1, 8, 8), dtype=np.uint8)
    b = a + 1                      # MSE = 1  ->  PSNR = 10*log10(255^2)
    assert abs(psnr(a, b) - 48.1308) < 1e-3


def test_snr_known_value():
    a = np.full(100, 100)
    b = a + 10                     # signal power 10000, noise power 100 -> 20 dB
    assert abs(snr(a, b) - 20.0) < 1e-9


def test_bit_error_rate():
    assert bit_error_rate([0, 1, 1, 0], [0, 1, 0, 1]) == 0.5
    assert bit_error_rate([], []) == 0.0
