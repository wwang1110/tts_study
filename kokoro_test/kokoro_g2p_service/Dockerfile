# Multi-stage Docker build for G2P Service
# This isolates GPL-licensed dependencies in a container

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies for espeak-ng and misaki
RUN apt-get update && apt-get install -y \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# First, install the CPU-only version of torch to minimize image size
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install spaCy English model to avoid runtime download conflicts
RUN python -m spacy download en_core_web_sm

# Copy the G2P service
COPY g2p_service.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash g2p
USER g2p

# Expose port
EXPOSE 5000

# Run the service with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--access-logfile", "-", "--error-logfile", "-", "g2p_service:app"]