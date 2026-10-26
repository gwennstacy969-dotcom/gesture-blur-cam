@echo off
title Gesture Blur Camera - Setup & Run
echo ============================================
echo   Gesture Blur Camera - Auto Setup
echo ============================================
echo.

echo [1/2] Memeriksa dependencies...
python -m pip install -r requirements.txt >nul 2>&1
echo [OK] Dependencies siap!
echo.

echo [2/2] Menjalankan Gesture Blur Camera...
echo.
python main.py

echo.
pause
