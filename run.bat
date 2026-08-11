@echo off
title Gesture Blur Camera - Setup & Run
echo ============================================
echo   Gesture Blur Camera - Auto Setup
echo ============================================
echo.

echo [1/2] Menginstall dependencies...
pip install --user opencv-python cvzone mediapipe >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] pip install --user gagal, mencoba tanpa --user...
    pip install opencv-python cvzone mediapipe >nul 2>&1
)
echo [OK] Dependencies siap!
echo.

echo [2/2] Menjalankan Gesture Blur Camera...
echo.
python main.py

echo.
pause
