@echo off
title Bimanual UR10e Robotic Web Platform
echo ================================================================================
echo       LAUNCHING BIMANUAL UR10e ROBOTIC WEB PLATFORM (FULLSTACK PLATFORM)
echo ================================================================================
if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe launch_platform.py %*
) else (
    python launch_platform.py %*
)
pause
