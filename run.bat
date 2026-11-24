@echo off
cd /d %~dp0
where ffmpeg >nul 2>nul || (echo FFmpeg not found. Install it and add it to PATH. & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -q -r requirements.txt
python app.py
