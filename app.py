"""Flash & Audio Layer Extractor — local web app.

Run:  python app.py   ->  opens http://localhost:8000
"""
import json
import shutil
import subprocess
import threading
import uuid
import webbrowser
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.io import wavfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "output"
STATIC = BASE / "static"
OUTPUT.mkdir(exist_ok=True)

ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm"}

# ---------------------------------------------------------------- settings
VISUAL = {
    #          min diff vs. both neighbours (0-255 scale), max (before/after diff ÷ flash diff)
    "low":    {"thr": 45.0, "ratio": 0.30},
    "medium": {"thr": 30.0, "ratio": 0.40},
    "high":   {"thr": 18.0, "ratio": 0.50},
}
AUDIO = {
    "low":    {"spike_db": 14, "flux_z": 6.0, "band_db": 18, "quiet_db": 35, "max_events": 30},
    "medium": {"spike_db": 10, "flux_z": 4.5, "band_db": 13, "quiet_db": 28, "max_events": 60},
    "high":   {"spike_db": 7,  "flux_z": 3.5, "band_db": 9,  "quiet_db": 20, "max_events": 120},
}
MAX_FLASH_FRAMES = 4
N_FFT, HOP = 2048, 512

RUNS: dict = {}  # run_id -> status dict


# ---------------------------------------------------------------- helpers
def fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"


def file_ts(seconds: float) -> str:
    return fmt_ts(seconds).replace(":", "m").replace(".", "s") + "ms"


def update(run_id, **kw):
    RUNS[run_id].update(kw)


# ---------------------------------------------------------------- visual
def _small(frame):
    s = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(s, (3, 3), 0).astype(np.float32)


def _diff(a, b):
    return float(np.mean(np.abs(a - b)))


def scan_video(run_id, video: Path, out: Path, sens: str):
    cfg = VISUAL[sens]
    frames_dir = out / "frames"
    thumbs_dir = frames_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0 or fps > 1000:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    buf = deque(maxlen=MAX_FLASH_FRAMES + 2)  # (idx, small, full)
    last_flag_end = -1
    results = []
    idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        buf.append((idx, _small(frame), frame))
        if total and idx % 25 == 0:
            update(run_id, progress=0.05 + 0.55 * min(idx / total, 1.0),
                   stage=f"Scanning frames {idx}/{total}")

        for k in range(1, MAX_FLASH_FRAMES + 1):
            if len(buf) < k + 2:
                break
            before, after = buf[-k - 2], buf[-1]
            mids = list(buf)[-k - 1:-1]
            if before[0] <= last_flag_end:
                continue
            m_min = min(min(_diff(m[1], before[1]), _diff(m[1], after[1])) for m in mids)
            if m_min < cfg["thr"]:
                continue
            base = _diff(before[1], after[1])
            if base > cfg["ratio"] * m_min:
                continue
            # flagged: k-frame outlier that returns to the surrounding content
            flash_ms = k * 1000.0 / fps
            for m_idx, _, m_full in mids:
                t = m_idx / fps
                name = f"{file_ts(t)}_f{m_idx:07d}"
                cv2.imwrite(str(frames_dir / f"{name}.png"), m_full)
                h, w = m_full.shape[:2]
                tw = 320
                thumb = cv2.resize(m_full, (tw, max(1, int(h * tw / w))), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(thumbs_dir / f"{name}.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                results.append({
                    "frame": m_idx,
                    "time": round(t, 4),
                    "timestamp": fmt_ts(t),
                    "flash_frames": k,
                    "flash_ms": round(flash_ms, 1),
                    "image": f"/output/{run_id}/frames/{name}.png",
                    "thumb": f"/output/{run_id}/frames/thumbs/{name}.jpg",
                })
            last_flag_end = mids[-1][0]
            break

    cap.release()
    return results, fps, idx + 1


# ---------------------------------------------------------------- audio
def has_audio(video: Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def extract_audio(video: Path, wav: Path):
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "2", "-ar", "44100",
         "-c:a", "pcm_s16le", str(wav)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("FFmpeg audio extraction failed: " + r.stderr[-500:])


def write_wav(path: Path, sr: int, data: np.ndarray):
    wavfile.write(str(path), sr, (np.clip(data, -1, 1) * 32767).astype(np.int16))


def make_stem(stereo: np.ndarray, sr: int):
    """Center-removed (side) channel with upward compression so quiet layers become audible."""
    left, right = stereo[:, 0], stereo[:, 1]
    mid = (left + right) / 2
    side = (left - right) / 2
    mid_rms = np.sqrt(np.mean(mid ** 2)) + 1e-12
    side_rms = np.sqrt(np.mean(side ** 2))
    if side_rms > 0.01 * mid_rms:
        src, mode = side, "center-removed (L−R), boosted"
    else:
        src, mode = mid, "mono source: upward-compressed full mix (no side channel present)"
    env = np.sqrt(ndimage.uniform_filter1d(src.astype(np.float64) ** 2, size=max(1, int(0.05 * sr))))
    gain = np.clip((0.1 / (env + 1e-9)) ** 0.6, 1.0, 100.0)  # up to +40 dB on quiet passages
    y = src * gain
    peak = np.max(np.abs(y)) or 1.0
    return (y / peak * 0.89).astype(np.float32), mode, side


def frame_features(mono: np.ndarray, side: np.ndarray, sr: int):
    """Per-hop features computed in chunks: rms dB, side rms dB, low/high band dB, spectral flux."""
    n = max(0, (len(mono) - N_FFT) // HOP + 1)
    win = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    low = (freqs >= 15) & (freqs < 60)
    nyq = sr / 2
    high = freqs >= min(15000, 0.7 * nyq)
    out = {k: np.empty(n, np.float32) for k in ("rms", "side", "low", "high", "flux")}
    prev = None
    chunk = 4096
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        idx = np.arange(c0, c1)[:, None] * HOP + np.arange(N_FFT)[None, :]
        fr = mono[idx]
        sf = side[idx]
        out["rms"][c0:c1] = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-6)
        out["side"][c0:c1] = 20 * np.log10(np.sqrt(np.mean(sf ** 2, axis=1)) + 1e-6)
        mag = np.abs(np.fft.rfft(fr * win, axis=1)).astype(np.float32)
        pw = mag ** 2
        out["low"][c0:c1] = 10 * np.log10(pw[:, low].sum(1) + 1e-10)
        out["high"][c0:c1] = 10 * np.log10(pw[:, high].sum(1) + 1e-10)
        lm = np.log1p(mag)
        p = np.vstack([prev[None, :] if prev is not None else lm[:1], lm])
        out["flux"][c0:c1] = np.maximum(np.diff(p, axis=0), 0).sum(1)
        prev = lm[-1]
    return out


def _runs(mask, score, max_len, label, fps_a, min_len=1):
    lab, n = ndimage.label(mask)
    events = []
    for sl in ndimage.find_objects(lab):
        s = sl[0]
        length = s.stop - s.start
        if min_len <= length <= max_len:
            events.append({"start": s.start / fps_a, "end": (s.stop * HOP + N_FFT) / (HOP * fps_a),
                           "types": {label}, "score": float(score[s].max())})
    return events


def detect_audio_events(f, sr, cfg):
    fps_a = sr / HOP
    w_short = max(3, int(1.5 * fps_a)) | 1
    w_long = max(3, int(3.0 * fps_a)) | 1
    max_short = int(0.6 * fps_a)
    ev = []

    rms = f["rms"]
    med = ndimage.median_filter(rms, size=w_short, mode="nearest")
    ex = rms - med
    ev += _runs((ex > cfg["spike_db"]) & (rms > -60), ex, max_short, "amplitude spike", fps_a)

    flux = f["flux"]
    fmed = ndimage.median_filter(flux, size=w_short, mode="nearest")
    mad = ndimage.median_filter(np.abs(flux - fmed), size=w_short, mode="nearest") + 1e-6
    z = (flux - fmed) / (1.4826 * mad)
    ev += _runs((z > cfg["flux_z"]) & (rms > -60), z, max_short, "spectral spike", fps_a)

    for key, label in (("low", "low-frequency burst (15–60 Hz)"), ("high", "high-frequency burst (≥15 kHz)")):
        # band energies use few FFT bins -> smooth over ~60 ms to suppress bin variance
        b = ndimage.uniform_filter1d(f[key], size=max(1, int(0.06 * fps_a)), mode="nearest")
        bmed = ndimage.median_filter(b, size=w_short, mode="nearest")
        bex = b - bmed
        ev += _runs((bex > cfg["band_db"]) & (b - rms > -25) & (rms > -65), bex, max_short, label, fps_a)

    s = f["side"]
    smed = ndimage.median_filter(s, size=w_short, mode="nearest")
    sex = s - smed
    ev += _runs((sex > cfg["spike_db"]) & (s > -55), sex, max_short, "side-channel spike", fps_a)

    lmed = ndimage.median_filter(rms, size=w_long, mode="nearest")
    dip = lmed - rms
    ev += _runs((dip > cfg["quiet_db"]) & (lmed > -45), dip, int(1.5 * fps_a),
                "quiet isolated segment", fps_a, min_len=max(1, int(0.05 * fps_a)))

    # ignore onsets/offsets at the very start/end of the track
    total = len(rms) * HOP / sr
    ev = [e for e in ev if e["end"] > 0.3 and e["start"] < total - 0.3]

    # merge overlaps / near-neighbours
    ev.sort(key=lambda e: e["start"])
    merged = []
    for e in ev:
        if merged and e["start"] <= merged[-1]["end"] + 0.15:
            m = merged[-1]
            m["end"] = max(m["end"], e["end"])
            m["types"] |= e["types"]
            m["score"] = max(m["score"], e["score"])
        else:
            merged.append(dict(e))
    merged.sort(key=lambda e: (len(e["types"]), e["score"]), reverse=True)
    merged = sorted(merged[: cfg["max_events"]], key=lambda e: e["start"])
    return merged


def save_spectrogram(clip: np.ndarray, sr: int, path: Path, t0: float, ev_start: float, ev_end: float):
    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=100)
    nfft = 1024
    ax.specgram(clip + 1e-9, NFFT=nfft, Fs=sr, noverlap=nfft * 3 // 4, cmap="magma",
                xextent=(t0, t0 + len(clip) / sr), vmin=-140)
    ax.set_yscale("symlog", linthresh=200)
    ax.set_ylim(20, sr / 2)
    ax.axvline(ev_start, color="cyan", lw=0.8, ls="--")
    ax.axvline(ev_end, color="cyan", lw=0.8, ls="--")
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("Hz", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def scan_audio(run_id, video: Path, out: Path, sens: str):
    adir = out / "audio"
    cdir = adir / "clips"
    cdir.mkdir(parents=True, exist_ok=True)
    if not has_audio(video):
        return {"present": False, "events": []}

    update(run_id, progress=0.62, stage="Extracting audio track (FFmpeg)")
    full = adir / "full_audio.wav"
    extract_audio(video, full)
    sr, data = wavfile.read(str(full))
    stereo = data.astype(np.float32) / 32768.0
    if stereo.ndim == 1:
        stereo = np.stack([stereo, stereo], 1)
    mono = stereo.mean(1)

    update(run_id, progress=0.67, stage="Building background / center-removed stem")
    stem, stem_mode, side = make_stem(stereo, sr)
    write_wav(adir / "background_stem.wav", sr, stem)

    update(run_id, progress=0.72, stage="Analysing audio")
    if len(mono) < N_FFT:
        return {"present": True, "events": [], "full": f"/output/{run_id}/audio/full_audio.wav",
                "stem": f"/output/{run_id}/audio/background_stem.wav", "stem_mode": stem_mode}
    feats = frame_features(mono, side, sr)
    events = detect_audio_events(feats, sr, AUDIO[sens])

    results = []
    pad, min_len = 0.4, 1.0
    dur = len(mono) / sr
    for i, e in enumerate(events):
        update(run_id, progress=0.75 + 0.24 * i / max(1, len(events)),
               stage=f"Exporting audio clips {i + 1}/{len(events)}")
        c0 = max(0.0, e["start"] - pad)
        c1 = min(dur, max(e["end"] + pad, c0 + min_len))
        a, b = int(c0 * sr), int(c1 * sr)
        name = f"{file_ts(e['start'])}_{i:03d}"
        write_wav(cdir / f"{name}.wav", sr, stereo[a:b])
        write_wav(cdir / f"{name}_stem.wav", sr, stem[a:b])
        save_spectrogram(mono[a:b], sr, cdir / f"{name}.png", c0, e["start"], e["end"])
        results.append({
            "start": round(e["start"], 4),
            "end": round(e["end"], 4),
            "timestamp": fmt_ts(e["start"]),
            "end_timestamp": fmt_ts(e["end"]),
            "duration_ms": round((e["end"] - e["start"]) * 1000, 1),
            "types": sorted(e["types"]),
            "clip": f"/output/{run_id}/audio/clips/{name}.wav",
            "stem_clip": f"/output/{run_id}/audio/clips/{name}_stem.wav",
            "spectrogram": f"/output/{run_id}/audio/clips/{name}.png",
        })
    return {"present": True, "events": results,
            "full": f"/output/{run_id}/audio/full_audio.wav",
            "stem": f"/output/{run_id}/audio/background_stem.wav", "stem_mode": stem_mode}


# ---------------------------------------------------------------- job
def run_job(run_id, video: Path, sens: str):
    out = video.parent
    try:
        update(run_id, status="running", progress=0.02, stage="Opening video")
        frames, fps, n = scan_video(run_id, video, out, sens)
        update(run_id, frames=frames, fps=round(fps, 3), frame_count=n)
        audio = scan_audio(run_id, video, out, sens)
        update(run_id, audio=audio, status="done", progress=1.0, stage="Done")
        (out / "results.json").write_text(json.dumps(RUNS[run_id], indent=2))
    except Exception as exc:  # surface error to UI
        update(run_id, status="error", stage="Error", error=str(exc))


# ---------------------------------------------------------------- web
app = FastAPI(title="Flash & Audio Layer Extractor")
app.mount("/output", StaticFiles(directory=OUTPUT), name="output")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/scan")
def scan(file: UploadFile = File(...), sensitivity: str = Form("medium")):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use .mp4, .mov, .mkv or .webm.")
    sens = sensitivity.lower() if sensitivity.lower() in VISUAL else "medium"
    run_id = uuid.uuid4().hex[:10]
    out = OUTPUT / run_id
    out.mkdir(parents=True)
    video = out / f"source{ext}"
    with video.open("wb") as fh:
        shutil.copyfileobj(file.file, fh, length=1024 * 1024)
    RUNS[run_id] = {"run_id": run_id, "filename": file.filename, "sensitivity": sens,
                    "status": "queued", "progress": 0.0, "stage": "Queued"}
    threading.Thread(target=run_job, args=(run_id, video, sens), daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/status/{run_id}")
def status(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Unknown run")
    return JSONResponse(RUNS[run_id])


if __name__ == "__main__":
    import uvicorn

    url = "http://localhost:8000"
    print(f"\n  Open {url}\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
