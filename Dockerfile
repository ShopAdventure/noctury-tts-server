# =============================================================================
# Noctury TTS Server — Qwen3-TTS Self-Hosted
# Multi-stage Docker build optimized for RunPod Serverless
# =============================================================================

# Stage 1: Builder - Install dependencies and download models
# =============================================================================
ARG DOCKER_FROM=nvidia/cuda:12.8.0-runtime-ubuntu22.04
FROM ${DOCKER_FROM} AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    build-essential \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsndfile1-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.8
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install flash-attention from pre-built wheel
RUN pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.6.3+cu128torch2.10-cp310-cp310-linux_x86_64.whl || true

# Pre-download models to cache them in the image
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-Tokenizer-12Hz')"
RUN python -c "import whisper; whisper.load_model('base')"

# =============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# =============================================================================
FROM ${DOCKER_FROM} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="Noctury / ShopAdventure"
LABEL description="Noctury TTS Server — Qwen3-TTS Self-Hosted with Voice Cloning"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsndfile1 \
    libmagic1 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && git lfs install

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy cached models from builder
COPY --from=builder /root/.cache /root/.cache

ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1

# Noctury default configuration
ENV DEFAULT_LANGUAGE=French
ENV DEFAULT_SPEED=0.92
ENV MAX_NEW_TOKENS=4096
ENV CHUNK_MAX_CHARS=400
ENV CROSSFADE_MS=200

# Create necessary directories
RUN mkdir -p /app/server/outputs /app/server/resources

# Copy server files
COPY server.py /app/server/
COPY start.sh /app/server/

# Copy voice reference files if they exist
COPY resources/ /app/server/resources/

# Fix line endings and make executable
RUN sed -i 's/\r$//' /app/server/start.sh \
    && chmod +x /app/server/start.sh

WORKDIR /app/server

# Expose the server port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "/app/server/start.sh"]
