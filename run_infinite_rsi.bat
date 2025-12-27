@echo off
setlocal
cd /d "%~dp0"
title RSI Infinite Loop Monitor (L2 Meta-Logic)

echo ====================================================
echo Starting Infinite Recursive Self-Improvement Loop...
echo ====================================================
echo Debug Info:
echo Current Directory: %cd%
if exist "L2_UNIFIED_RSI.py" (
    echo Code File Found: Yes
) else (
    echo Code File Found: NO
    echo [CRITICAL ERROR] L2_UNIFIED_RSI.py is missing!
    echo You must download L2_UNIFIED_RSI.py to this folder.
    pause
    exit /b
)
echo.

REM Try 'python' command
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Using command: python
    python L2_UNIFIED_RSI.py rsi-loop --generations 500 --rounds 100
    goto :Done
)

REM Try 'py' command (Windows Launcher)
py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Using command: py
    py L2_UNIFIED_RSI.py rsi-loop --generations 500 --rounds 100
    goto :Done
)

REM Try 'python3' command
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Using command: python3
    python3 L2_UNIFIED_RSI.py rsi-loop --generations 500 --rounds 100
    goto :Done
)

echo [ERROR] Python not found!
echo Tried: python, py, python3
echo Please install Python from python.org and check "Add to PATH".
pause
exit /b

:Done
echo.
echo RSI Loop stopped.
pause
