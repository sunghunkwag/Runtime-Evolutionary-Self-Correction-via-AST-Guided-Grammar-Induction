@echo off
setlocal
cd /d "%~dp0"
title RSI Infinite Loop Monitor (L2 Meta-Logic) - STATUS: STARTING

cls
echo ==========================================================
echo       RSI SELF-IMPROVEMENT ENGINE LAUNCHER V3
echo ==========================================================
echo.

REM 1. File Check
echo [1/3] Checking for L2_UNIFIED_RSI.py...
if exist "L2_UNIFIED_RSI.py" (
    echo    - OK: File found.
) else (
    color 4f
    echo.
    echo    [CRITICAL ERROR] FILE MISSING!
    echo    --------------------------------------------------
    echo    Could not find 'L2_UNIFIED_RSI.py' in this folder:
    echo    %cd%
    echo.
    echo    You must download BOTH the .bat and the .py file.
    echo    --------------------------------------------------
    echo.
    pause
    exit /b
)
echo.

REM 2. Python Check & Run
echo [2/3] Searching for Python...
for %%C in (python py python3) do (
    %%C --version >nul 2>&1
    if not errorlevel 1 (
        echo    - Found: %%C
        echo.
        echo [3/3] Starting Infinite Loop...
        title RSI Infinite Loop - RUNNING
        
        %%C L2_UNIFIED_RSI.py rsi-loop --generations 500 --rounds 100
        
        if errorlevel 1 (
            color 4f
            title RSI Infinite Loop - CRASHED
            echo.
            echo    [EXECUTION FAILED]
            echo    The Python script crashed with an error.
            echo    Please check the error message above.
            echo.
            pause
            exit /b
        )
        goto :Success
    )
)

REM 3. Python Failure
color 4f
echo.
echo    [CRITICAL ERROR] PYTHON NOT FOUND!
echo    --------------------------------------------------
echo    Tried commands: python, py, python3
echo    None of them worked. Please install Python from:
echo    https://www.python.org/downloads/
echo    (Make sure to check 'Add Python to PATH' during install)
echo    --------------------------------------------------
echo.
pause
exit /b

:Success
echo.
echo    [INFO] Process finished successfully.
pause
