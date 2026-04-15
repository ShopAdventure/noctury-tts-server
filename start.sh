#!/bin/bash

echo "=== Noctury TTS Server Starting ==="
echo "Device: $(python3 -c 'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")')"
echo "Language: ${DEFAULT_LANGUAGE:-French}"
echo "Speed: ${DEFAULT_SPEED:-0.92}"
echo "Max Tokens: ${MAX_NEW_TOKENS:-4096}"

# Change to the directory containing the server.py file
cd "$(dirname "$0")"

# Add the current directory to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Use RunPod PORT env var if available, otherwise default to 7860
SERVER_PORT=${PORT:-7860}
echo "Server port: $SERVER_PORT"

# Start the FastAPI server with uvicorn in the background
python3 -m uvicorn server:app --host 0.0.0.0 --port $SERVER_PORT --workers 1 &

# Keep container running for RunPod web terminal access
sleep infinity
