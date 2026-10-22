"""Embedding-strength profiles and their measured behaviour (from scripts/strength_sweep.py)."""

PROFILES = {
    "stealth": dict(audio=dict(delta=0.3, reps=1), video=dict(delta=12.0, reps=2)),
    "low": dict(audio=dict(delta=0.5, reps=1), video=dict(delta=24.0, reps=4)),
    "medium": dict(audio=dict(delta=0.8, reps=1), video=dict(delta=40.0, reps=8)),
    "robust": dict(audio=dict(delta=0.8, reps=2), video=dict(delta=64.0, reps=16)),
}

# Measured on real clips: quality (dB) and the average share of attacks survived (0..1).
# audio_rob: mean of MP3 128 and MP3 64.  video_rob: mean of H.264 CRF 23/28, H.265 CRF 28, resize.
MEASURED = {
    "stealth": dict(audio_snr=32.483, video_psnr=56.724, audio_rob=0.125, video_rob=0.0),
    "low": dict(audio_snr=28.688, video_psnr=47.998, audio_rob=0.5, video_rob=0.25),
    "medium": dict(audio_snr=24.297, video_psnr=40.530, audio_rob=0.75, video_rob=0.5),
    "robust": dict(audio_snr=21.533, video_psnr=33.282, audio_rob=1.0, video_rob=0.8125),
}