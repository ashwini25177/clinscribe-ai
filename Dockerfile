# ==============================================================================
# ClinScribe AI — Dockerfile
# ==============================================================================
# Multi-stage build for the clinical speech-to-insight inference server.

FROM python:3.11-slim AS base

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# ==============================================================================
# Production image
# ==============================================================================
FROM base AS production

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ASR_MODEL_PATH=/app/checkpoints/asr/final_model
ENV NER_MODEL_PATH=/app/checkpoints/ner/final_model
ENV SUMMARIZER_MODEL_PATH=/app/checkpoints/summarizer/final_model

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start the API server
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
