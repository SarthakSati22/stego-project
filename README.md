\# Adaptive Cross-Modal Steganography Against Multimodal Steganalysis



A secret message is encrypted (AES-GCM), protected with error-correction

coding, and split across audio and video carriers. The split ratio is

adapted using the detection risk reported by a joint multimodal

steganalyzer. The project studies whether this risk-aware allocation

improves security compared with fixed or single-carrier strategies.



\## Setup



Requires Python 3.11+ and FFmpeg (with libx264, libx265 and libmp3lame).



&#x20;   python -m venv .venv

&#x20;   .venv\\Scripts\\activate

&#x20;   pip install -r requirements.txt

&#x20;   python -m pytest -v



All three tests should pass.



\## Evaluation protocol



This protocol is fixed before running experiments and must match

`configs/default.yaml`. Do not change it after seeing results.



\- \*\*Data format:\*\* 10 s clips; audio 16 kHz, mono, 16-bit; video 256x256

&#x20; at 25 fps.

\- \*\*Data split:\*\* 70% train, 15% validation, 15% test, split by clip

&#x20; (never by frame), listed in `data/manifest.csv`.

\- \*\*Payload sizes:\*\* 32, 64 and 128 bytes before encryption and

&#x20; error-correction coding.

\- \*\*Attacks (real codecs via FFmpeg):\*\*

&#x20; - MP3 compression at 128 and 64 kbps

&#x20; - H.264 and H.265 compression at CRF 23 and 28

&#x20; - Frame dropping at 5% and 10%

&#x20; - Resizing to 75%

\- \*\*Strategies compared:\*\*

&#x20; 1. Audio-only

&#x20; 2. Video-only

&#x20; 3. Fixed 50:50 split

&#x20; 4. Adaptive allocation without detector feedback

&#x20; 5. Adaptive allocation with detector feedback

\- \*\*Metrics:\*\*

&#x20; - Video quality: PSNR, SSIM

&#x20; - Audio quality: SNR

&#x20; - Recovery: bit error rate, recovery rate

&#x20; - Detectability: detector accuracy and AUC

\- \*\*Seeds:\*\* each experiment is run with seeds 0, 1 and 2; results are

&#x20; reported as mean +/- standard deviation.

\- \*\*Negative results:\*\* a negative result (adaptive allocation not

&#x20; beating the baselines) is a valid outcome and will be reported.



\## Project layout



\- `src/crypto` - AES-GCM encryption and error-correction coding

\- `src/audio` - STFT-based audio embedder and extractor

\- `src/video` - transform-domain video embedder and extractor

\- `src/channel` - attacks (MP3, H.264/H.265, frame drop, resize)

\- `src/steganalysis` - audio, video and fusion detectors

\- `src/allocator` - payload allocation logic

\- `src/eval` - metrics and plotting

\- `configs/` - experiment parameters (YAML)

\- `experiments/` - one script per experiment

\- `tests/` - unit tests

\- `data/` and `results/` - not committed to Git



\## Running tests



&#x20;   python -m pytest -v

