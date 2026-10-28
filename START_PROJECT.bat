@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo Starting RoomSense - Room Occupancy Estimation...

if not exist ".venv\Scripts\python.exe" (
    echo [.venv not found] Running initial setup first...
    call SETUP.bat
    if not exist ".venv\Scripts\python.exe" (
        echo [ERROR] Virtual environment could not be set up. Please run SETUP.bat manually.
        pause
        exit /b 1
    )
)

if not exist "artifacts\occupancy_model.joblib" (
    echo [Model artifact missing] Training the model...
    ".venv\Scripts\python.exe" train.py
)

".venv\Scripts\python.exe" scripts\launcher.py
if errorlevel 1 pause
