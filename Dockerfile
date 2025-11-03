# ==========================================================
# UNIVERSAL DOCKERFILE FOR FASTAPI + CELERY + WHISPER.CPP
# Builds whisper.cpp from source (robust & future-proof)
# ==========================================================

# -------------------------
# Stage 1: Builder (compile whisper.cpp + install Python deps)
# -------------------------
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/home/appuser/.local/bin:$PATH \
    WHISPER_MODEL_PATH=/app/whisper.cpp/models/ggml-tiny.en-q8_0.bin \
    WHISPER_CLI_PATH=/usr/local/bin/whisper-cli

# ✅ FINAL FIX: Add a harmless comment to force a clean rebuild on Railway.
# This comment invalidates the cache and ensures the libraries below are installed.
# Version: 2

# Install runtime deps (including whisper.cpp dependencies) and create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    wget \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd --no-log-init -r -g appuser -m -d /home/appuser appuser

# Set the working directory
WORKDIR /app

# Copy requirements file and set ownership
COPY --chown=appuser:appuser requirements.txt .

# Switch to the non-root user before running pip
USER appuser

# Run pip install as the 'appuser'
RUN pip install --user --no-cache-dir -r requirements.txt

# Switch back to root to perform system-level tasks like git cloning
USER root

# Clone whisper.cpp and build it from source
RUN git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git /app/whisper.cpp && \
    cd /app/whisper.cpp && \
    rm -rf build && mkdir -p build && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON && \
    cmake --build build --config Release -j$(nproc)

# -------------------------
# Stage 2: Runtime image (smaller)
# -------------------------
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/home/appuser/.local/bin:$PATH \
    WHISPER_MODEL_PATH=/app/whisper.cpp/models/ggml-tiny.en-q8_0.bin \
    WHISPER_CLI_PATH=/usr/local/bin/whisper-cli

# Install runtime deps (including whisper.cpp dependencies) and create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    wget \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd --no-log-init -r -g appuser -m -d /home/appuser appuser

# Set working directory
WORKDIR /app

# Change the ownership of the working directory to the appuser
# This allows the user to create files like the celerybeat-schedule db.
RUN chown appuser:appuser /app

# Copy Python packages installed in builder and set ownership
COPY --from=builder --chown=appuser:appuser /home/appuser/.local /home/appuser/.local

# Copy whisper.cpp artefacts (models + binary)
COPY --from=builder /app/whisper.cpp /app/whisper.cpp

# Put the binary on the system PATH once, for backend/worker/beat
RUN cp /app/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli && \
    chmod +x /usr/local/bin/whisper-cli && \
    ln -sf /usr/local/bin/whisper-cli /usr/local/bin/whisper

# Download a small default model
RUN mkdir -p /app/whisper.cpp/models && \
    wget -q -O ${WHISPER_MODEL_PATH} \
      https://huggingface.co/ggml-org/whisper.cpp/resolve/main/ggml-tiny.en-q8_0.bin || \
    (echo "WARNING: failed to download model; continue without model" >&2)

# Always use latest yt-dlp
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
    -o /usr/local/bin/yt-dlp && \
    chmod a+rx /usr/local/bin/yt-dlp

# Copy project files and set ownership
COPY --chown=appuser:appuser . .

# Switch to the non-root user for the final runtime environment
USER appuser

# Expose FastAPI port
EXPOSE 5000

# Default command (will run as 'appuser')
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]

