"""
Pre-download models at Docker build time for fast cold start.
Called during: RUN python3 download_models.py
"""
from huggingface_hub import snapshot_download
import whisper

models = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
]

for model in models:
    print(f"Pre-downloading {model}...")
    path = snapshot_download(model)
    print(f"  -> Ready at: {path}")

print("Pre-downloading Whisper base...")
whisper.load_model("base")
print("  -> Whisper ready")

print("All models pre-downloaded successfully.")
