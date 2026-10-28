@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo =======================================================
echo Setting up RoomSense - Room Occupancy Estimation
echo =======================================================

:: Locate Python
set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 --version >nul 2>nul && set "PY_CMD=py -3.12"
    if not defined PY_CMD (
        py -3 --version >nul 2>nul && set "PY_CMD=py -3"
    )
)
if not defined PY_CMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        python --version >nul 2>nul && set "PY_CMD=python"
    )
)
if not defined PY_CMD (
    where python3 >nul 2>nul
    if not errorlevel 1 (
        python3 --version >nul 2>nul && set "PY_CMD=python3"
    )
)

if not defined PY_CMD (
    echo [ERROR] Python was not found on your system PATH.
    echo Please install Python 3.10 or newer from https://www.python.org/
    echo Make sure to check the box "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [OK] Using Python command: %PY_CMD%

:: Create virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment in .venv...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists in .venv.
)

:: Install dependencies
echo [*] Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 (
    echo [*] Locked versions had a resolution issue; trying requirements.in...
    ".venv\Scripts\python.exe" -m pip install -r requirements.in
    if errorlevel 1 goto :failed
)
echo [OK] Dependencies installed successfully.

:: Model and data checks
if not exist "artifacts\occupancy_model.joblib" (
    echo [*] Model artifact not found. Training model now...
    ".venv\Scripts\python.exe" train.py
    if errorlevel 1 goto :failed
) else (
    echo [OK] Pre-trained model found in artifacts\occupancy_model.joblib.
)

echo.
echo =======================================================
echo [SUCCESS] RoomSense setup completed!
echo Double-click START_PROJECT.bat to launch RoomSense.
echo =======================================================
pause
exit /b 0

:failed
echo.
echo =======================================================
echo [ERROR] Setup encountered an issue. Please review the message above.
echo =======================================================
pause
exit /b 1
