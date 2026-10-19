@echo off
echo ── Starting Karen CCTV Tracking ──────────────────────────────────────────
echo   Cameras : feed/c1.mp4  feed/c2.mp4  feed/c3.mp4
echo   Map     : MAP.png  (config: map.json)
echo   Press Q in the video window to stop.
echo.

.\.venv\Scripts\python.exe main.py ^
    --source feed/c1.mp4 feed/c2.mp4 feed/c3.mp4 ^
    --map-config map.json ^
    --map-image MAP.png ^
    --threshold 0.7

pause
