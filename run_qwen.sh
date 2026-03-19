#!/bin/bash

# Configuration
venv_dir="venv_qwen"

echo "================================================="
echo "🦍 QWEN-32B LOCAL ENGINE INITIALIZATION"
echo "================================================="
echo "Note: If you run out of RAM, close Orpheus-TTS!"
echo ""

# Check if venv exists
if [ ! -d "$venv_dir" ]; then
    echo "[1] Creating virtual environment ($venv_dir)..."
    python3 -m venv "$venv_dir"
fi

# Activate venv
echo "[2] Activating virtual environment..."
source "$venv_dir/bin/activate"

# Install dependencies (will only install if missing/outdated)
echo "[3] Installing dependencies with Metal support..."
CMAKE_ARGS="-DGGML_METAL=on" pip install -r requirements.txt

# Run the server
echo "[4] Booting the Qwen-32B Model server..."
python local_llm_server.py
