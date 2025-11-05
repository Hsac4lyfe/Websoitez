# ==========================================================
# UNIVERSAL DOCKERFILE FOR FASTAPI + CELERY + WHISPER.CPP
# Final, Debugged Version
# ==========================================================

# -------------------------
# Stage 1: Builder (compile whisper.cpp, download model, install Python deps)
# -------------------------
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/home/appuser/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd --no-log-init -r -g appuser -m -d /home/appuser appuser

WORKDIR /app

# --- Python Dependencies ---
COPY --chown=appuser:appuser requirements.txt .
USER appuser
RUN pip install --user --no-cache-dir -r requirements.txt
USER root

# --- Build whisper.cpp from source ---
RUN git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git /app/whisper.cpp && \
    cd /app/whisper.cpp && \
    rm -rf build && mkdir -p build && \
    # Build with shared libraries, which is more standard
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON && \
    cmake --build build --config Release -j$(nproc)

# ✅ FINAL, CORRECT FIX: Download the model file reliably here in the builder stage.
RUN mkdir -p /app/whisper.cpp/models && \
    wget -q -O /app/whisper.cpp/models/ggml-tiny.en-q8_0.bin \
      https://huggingface.co/ggml-org/whisper.cpp/resolve/main/ggml-tiny.en-q8_0.bin

# -------------------------
# Stage 2: Runtime image (smaller)
# -------------------------
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/home/appuser/.local/bin:$PATH \
    WHISPER_MODEL_PATH=/app/whisper.cpp/models/ggml-tiny.en-q8_0.bin \
    WHISPER_CLI_PATH=/usr/local/bin/whisper-cli \
    # This provides the necessary runtime libraries for the shared build
    LD_LIBRARY_PATH=/app/whisper.cpp/build/lib

# Add a nudge comment to force a clean rebuild on Railway
# Version: 6

# Install runtime deps and create user
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd --no-log-init -r -g appuser -m -d /home/appuser appuser

WORKDIR /app
RUN chown appuser:appuser /app

# --- Copy Artifacts from Builder ---
COPY --from=builder --chown=appuser:appuser /home/appuser/.local /home/appuser/.local
# Copy the entire built whisper.cpp directory, which now INCLUDES the model
COPY --from=builder /app/whisper.cpp /app/whisper.cpp

# --- Set up Binaries ---
RUN cp /app/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli && \
    chmod +x /usr/local/bin/whisper-cli && \
    ln -sf /usr/local/bin/whisper-cli /usr/local/bin/whisper

RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
    -o /usr/local/bin/yt-dlp && \
    chmod a+rx /usr/local/bin/yt-dlp

# --- Final Setup ---
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 5000
CMD ["uvicorn", "main.app", "--host", "0.0.0.0", "--port", "5000"]
