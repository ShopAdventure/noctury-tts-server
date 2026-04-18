#!/bin/bash
set -e

echo "=== Noctury TTS Jobs Handler Starting ==="
python3 -c "import torch; print('Device: CUDA' if torch.cuda.is_available() else 'Device: CPU')" 2>/dev/null || true

VOLUME_MODELS="/runpod-volume/models"
if [ -d "$VOLUME_MODELS/Qwen3-TTS-12Hz-1.7B-VoiceDesign" ]; then
    echo "Network Volume detected — using pre-loaded models"
    export NOCTURY_MODEL_VOICEDESIGN="$VOLUME_MODELS/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
else
    echo "No pre-loaded models found, will download on first use"
fi

echo "Starting RunPod Jobs Handler..."
exec python3 /app/server/handler.py
