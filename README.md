# Flash & Audio Extractor

**Phone version:** `mobile/flash-finder.html` is a single-page version that runs entirely in the browser (iPhone, Android or desktop) with no install and no server. The video never leaves the device. It uses the same detection rules and thresholds as the Python app below.

A local web app that scans a video for brief visual flashes and short or unusual audio moments. It shows only the extracted frames, the audio clips, spectrograms and exact timestamps.

## Run

Requires Python 3.10+ and FFmpeg (`ffmpeg` and `ffprobe`) on your PATH.

```bash
./run.sh          # macOS / Linux
run.bat           # Windows
# or manually:
pip install -r requirements.txt
python app.py
```

The server starts on http://localhost:8000 and opens your browser.

## What it does

**Images.** Every frame is read at the video's native frame rate. A run of 1 to 4 frames is flagged when each of those frames differs sharply from the frame before the run and from the frame after it, while the frames before and after are similar to each other. Because of that last condition, ordinary scene cuts are not flagged. Each flagged frame is saved as `output/<run_id>/frames/<MMmSSsmmmms>_f<frame>.png`.

**Audio.** FFmpeg extracts the track to `output/<run_id>/audio/full_audio.wav`. The app then flags these events:
- amplitude spikes
- spectral-flux spikes
- low-frequency bursts (15–60 Hz)
- high-frequency bursts (≥15 kHz)
- spikes in the side channel
- short quiet segments surrounded by louder audio

It also writes these files:
- `background_stem.wav`: the center-removed (L−R) channel, boosted with upward compression. If the source is mono, it uses an upward-compressed full mix instead.
- `clips/<timestamp>_NNN.wav` (original), `_stem.wav` (stem) and `.png` (spectrogram) for each flagged moment. Dashed lines on the spectrogram mark the flagged span.

**Sensitivity.** Low, Medium and High set the detection thresholds. High flags more candidates.
