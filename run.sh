#!/usr/bin/env bash
# Creates a venv on first run, installs deps, starts the server and opens http://localhost:8000
set -e
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo "FFmpeg not found. Install it first (brew install ffmpeg / apt install ffmpeg)."; exit 1; }
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
python app.py
