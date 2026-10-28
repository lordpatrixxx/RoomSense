#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== Setting up RoomSense - Room Occupancy Estimation ==="

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip --quiet

echo "Installing requirements..."
pip install -r requirements-lock.txt || pip install -r requirements.txt

if [ ! -f "artifacts/occupancy_model.joblib" ]; then
    echo "Training model..."
    python train.py
fi

echo "=== Setup complete! Run ./run.sh to start RoomSense ==="
