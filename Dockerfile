# =============================================================================
# Noctury TTS Server — Qwen3-TTS Self-Hosted
# Lightweight Docker build — models are downloaded at first startup
# Optimized for RunPod Serverless (smaller image = faster pull)
# =============================================================================

ARG DOCKER_FROM=nvidia/cuda:12.8.0-runtime-ubuntu22.04
FROM ${DOCKER_FROM}

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="Noctury / ShopAdventure"
LABEL description="Noctury TTS Server — Qwen3-TTS Self-Hosted with Voice Cloning"

# Install all dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    git-lfs \
    build-essential \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsndfile1-dev \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && git lfs install

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

# Try to install flash-attention (optional, non-blocking)
RUN pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.6.3+cu128torch2.10-cp310-cp310-linux_x86_64.whl || true

# NOTE: Models are NOT downloaded during build to keep the image small (~5GB)
# They will be downloaded at first startup via start.sh

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

# Expose the server port (RunPod will use PORT env var)
EXPOSE 7860

# Health check on dynamic port
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:${PORT:-7860}/ping || exit 1

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "/app/server/start.sh"]
