# Multi-stage Docker build for TTS Service
# License-safe TTS service that communicates with GPL-isolated G2P service

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base

# Set working directory
WORKDIR /app

# Copy cache directory
COPY ./.cache /app/.cache
COPY ./kokoro /app/kokoro
COPY ./tts_service.py /app/tts_service.py

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
ENV FIRST_CHUNK_MAX_TOKENS=10
ENV FIRST_CHUNK_MIN_TOKENS=5
ENV HOST=0.0.0.0
ENV PORT=8880

# Run the service with Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8880", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "tts_service:app"]