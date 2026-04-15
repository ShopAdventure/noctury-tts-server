# =============================================================================
# Noctury TTS Server — Qwen3-TTS Self-Hosted
# Optimized for RunPod Serverless Load Balancing
# Base: nvidia/cuda (lightweight) + PyTorch via pip
# Models downloaded at first startup (not baked into image)
# =============================================================================

FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="Noctury / ShopAdventure"
LABEL description="Noctury TTS Server — Qwen3-TTS with Voice Cloning"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1-dev \
    libmagic1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# CUDA compat
RUN ldconfig /usr/local/cuda-12.1/compat/

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Try flash-attention (optional, non-blocking)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || true

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

# Copy voice reference files (Maxime voice sample for cloning)
COPY resources/maxime.mp3 /app/server/resources/maxime.mp3
COPY resources/.gitkeep /app/server/resources/.gitkeep

# Fix line endings and make executable
RUN sed -i 's/\r$//' /app/server/start.sh \
    && chmod +x /app/server/start.sh

WORKDIR /app/server

# Default port for RunPod Load Balancer
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:${PORT:-8000}/ping || exit 1

ENTRYPOINT ["/bin/bash", "/app/server/start.sh"]
