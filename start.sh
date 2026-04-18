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

# ─── Network Volume: use pre-loaded models if available ───────────────────────
VOLUME_MODELS="/runpod-volume/models"

if [ -d "$VOLUME_MODELS/Qwen3-TTS-12Hz-1.7B-Base" ] && [ -d "$VOLUME_MODELS/Qwen3-TTS-12Hz-1.7B-VoiceDesign" ]; then
    echo "=== Network Volume detected — using pre-loaded models (fast cold start) ==="
    export NOCTURY_MODEL_BASE="$VOLUME_MODELS/Qwen3-TTS-12Hz-1.7B-Base"
    export NOCTURY_MODEL_VOICEDESIGN="$VOLUME_MODELS/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    export NOCTURY_MODEL_COSYVOICE="$VOLUME_MODELS/CosyVoice2-0.5B"
    echo "  Base model: $NOCTURY_MODEL_BASE"
    echo "  VoiceDesign model: $NOCTURY_MODEL_VOICEDESIGN"
    echo "  CosyVoice2: $NOCTURY_MODEL_COSYVOICE"
else
    echo "=== No network volume — downloading models from HuggingFace ==="
    python3 /app/server/download_models.py
fi

echo "=== Models ready ==="

# Use RunPod PORT env var if available, otherwise default to 8000
SERVER_PORT=${PORT:-8000}
echo "Server port: $SERVER_PORT"

# ─── Mode dual : FastAPI (routes courtes) + RunPod Jobs handler (générations longues) ───
# RUNPOD_JOBS_MODE=1 → démarre uniquement le Jobs handler (sans FastAPI)
# RUNPOD_JOBS_MODE=0 ou absent → mode normal FastAPI uniquement
# RUNPOD_DUAL_MODE=1 → démarre les deux en parallèle

if [ "${RUNPOD_JOBS_MODE}" = "1" ]; then
    echo "=== Mode RunPod Jobs Handler ==="
    echo "Démarrage du handler asynchrone (sans timeout de proxy)..."
    python3 /app/server/handler.py
elif [ "${RUNPOD_DUAL_MODE}" = "1" ]; then
    echo "=== Mode Dual : FastAPI + RunPod Jobs Handler ==="
    # Démarrer FastAPI en arrière-plan
    python3 -m uvicorn server:app --host 0.0.0.0 --port $SERVER_PORT --workers 1 &
    FASTAPI_PID=$!
    echo "FastAPI démarré (PID $FASTAPI_PID) sur port $SERVER_PORT"
    # Démarrer le Jobs handler en avant-plan
    python3 /app/server/handler.py
else
    echo "=== Mode FastAPI standard ==="
    # Start the FastAPI server with uvicorn in the background
    python3 -m uvicorn server:app --host 0.0.0.0 --port $SERVER_PORT --workers 1 &
    # Keep container running for RunPod web terminal access
    sleep infinity
fi
