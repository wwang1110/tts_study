# Multi-stage Docker build for G2P Service
# This isolates GPL-licensed dependencies in a container

FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies for espeak-ng and misaki
RUN apt-get update && apt-get install -y \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the G2P service
COPY g2p_service.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash g2p
USER g2p

# Expose port
EXPOSE 5000

# Run the service with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--access-logfile", "-", "--error-logfile", "-", "g2p_service:app"]