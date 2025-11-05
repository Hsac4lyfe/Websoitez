# ==========================================================
# UNIVERSAL DOCKERFILE FOR FASTAPI + CELERY + WHISPER.CPP
# Builds whisper.cpp from source (robust & future-proof)
# ==========================================================

# -------------------------
# Stage 1: Builder (compile whisper.cpp + install Python deps)
# -------------------------
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    # Correct path for the non-root user's home directory
    PATH=/home/appuser/.local/bin:$PATH

# Install build tools and create a non-root user
# This is done as root (the default user)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ffmpeg \
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
    # ✅ FINAL, CORRECT FIX: Build a self-contained, static executable.
    # This removes the need for any external library "bolts".
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build --config Release -j$(nproc)

# -------------------------
# Stage 2: Runtime image (smaller)
# -------------------------
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/home/appuser/.local/bin:$PATH \
    WHISPER_MODEL_PATH=/app/whisper.cpp/models/ggml-tiny.en-q8_0.bin \
    WHISPER_CLI_PATH=/usr/local/bin/whisper-cli

# Add a nudge comment to force a clean rebuild on Railway.
# Version: 5

# Install only the absolute essential runtime deps and create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd --no-log-init -r -g appuser -m -d /home/appuser appuser

# Set working directory
WORKDIR /app

# Change the ownership of the working directory to the appuser
RUN chown appuser:appuser /app

# Copy Python packages installed in builder and set ownership
COPY --from=builder --chown=appuser:appuser /home/appuser/.local /home/appuser/.local

# Copy only the self-contained whisper-cli binary and its models.
# We no longer need the rest of the build artifacts.
COPY --from=builder /app/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=builder /app/whisper.cpp/models /app/whisper.cpp/models

# Put the binary on the system PATH once, for backend/worker/beat
RUN chmod +x /usr/local/bin/whisper-cli && \
    ln -sf /usr/local/bin/whisper-cli /usr/local/bin/whisper

# Download a small default model (if not already copied)
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
