"""Quality, recovery and detection metrics."""
import cv2
import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio (dB) for 8-bit video."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def _ssim_frame(x: np.ndarray, y: np.ndarray) -> float:
    """Standard SSIM (Gaussian 11x11 window, sigma 1.5) for one 8-bit frame."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    x, y = x.astype(np.float64), y.astype(np.float64)

    def blur(img):
        return cv2.GaussianBlur(img, (11, 11), 1.5)

    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cxy = blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return float(s[5:-5, 5:-5].mean())          # ignore the border


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity over frames."""
    return float(np.mean([_ssim_frame(x, y) for x, y in zip(a, b)]))


def snr(a: np.ndarray, b: np.ndarray) -> float:
    """Signal-to-noise ratio (dB) of audio b relative to the original a."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    noise = np.sum((a - b) ** 2)
    return float("inf") if noise == 0 else float(10 * np.log10(np.sum(a ** 2) / noise))


def bit_error_rate(sent, received) -> float:
    sent, received = np.asarray(sent), np.asarray(received)
    return float(np.mean(sent != received)) if len(sent) else 0.0