# Multi-stage Docker build for TTS Service
# License-safe TTS service that communicates with GPL-isolated G2P service

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy cache directory
COPY ./.cache /app/.cache
COPY ./kokoro /app/kokoro
COPY ./tts_service.py /app/tts_service.py

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash tts
RUN chown -R tts:tts /app
USER tts

# Expose port (default 8880)
EXPOSE 8880

# Run the service with Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8880", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "tts_service:app"]