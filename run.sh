#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

streamlit run app.py
