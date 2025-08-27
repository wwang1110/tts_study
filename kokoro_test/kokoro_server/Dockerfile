# Multi-stage Docker build for TTS Service
# License-safe TTS service that communicates with GPL-isolated G2P service

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing and PyTorch
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the TTS service files
COPY tts_service.py .
COPY safe_pipeline.py .

# Create directory for Kokoro models (will be mounted or downloaded)
RUN mkdir -p /app/models

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash tts
RUN chown -R tts:tts /app
USER tts

# Expose port (default 8880)
EXPOSE 8880

# Environment variables with defaults
ENV G2P_SERVICE_URL=http://0.0.0.0:5000
ENV G2P_TIMEOUT=30
ENV MAX_BATCH_SIZE=50
ENV MAX_TOKENS_PER_CHUNK=400
ENV MIN_TOKENS_PER_CHUNK=100
ENV FIRST_CHUNK_MAX_TOKENS=50
ENV FIRST_CHUNK_MIN_TOKENS=25
ENV HOST=0.0.0.0
ENV PORT=8880

# Run the service with Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8880", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "tts_service:app"]