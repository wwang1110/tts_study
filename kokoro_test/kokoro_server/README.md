# Kokoro TTS Server

A configurable FastAPI-based TTS service using Kokoro models with GPL-isolated G2P service integration.

## Features

- **Configurable G2P endpoint** - Use any compatible G2P service
- **Smart text chunking** - Optimal text splitting for streaming
- **Multiple TTS modes** - Single, batch, and streaming TTS
- **License compliance** - GPL-isolated architecture
- **Pause control** - Support for `[pause:1.5s]` tags
- **Multiple formats** - WAV and PCM audio output

## Configuration

The service is configured via environment variables. Copy `.env.example` to `.env` and modify as needed.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `G2P_SERVICE_URL` | `http://localhost:5000` | URL of the G2P service |
| `G2P_TIMEOUT` | `30` | Timeout for G2P requests (seconds) |
| `MAX_BATCH_SIZE` | `50` | Maximum requests per batch |
| `MAX_TOKENS_PER_CHUNK` | `400` | Maximum tokens per text chunk |
| `MIN_TOKENS_PER_CHUNK` | `100` | Minimum tokens per text chunk |
| `HOST` | `0.0.0.0` | Server host address |
| `PORT` | `8880` | Server port |

### G2P Service Configuration

The service requires a compatible G2P service running at the configured endpoint. The G2P service should:

- Accept POST requests to `/convert` with JSON: `{"text": "...", "lang": "..."}`
- Return JSON with: `{"phonemes": "..."}`
- Provide a `/health` endpoint for health checks

#### Example G2P Service URLs:

```bash
# Local development
export G2P_SERVICE_URL=http://localhost:5000

# Docker Compose
export G2P_SERVICE_URL=http://g2p-service:5000

# Remote service
export G2P_SERVICE_URL=https://g2p.example.com
export G2P_TIMEOUT=60
```

## Quick Start

### 1. Set up environment
```bash
# Copy example configuration
cp .env.example .env

# Edit configuration as needed
nano .env
```

### 2. Start the service
```bash
# Install dependencies
pip install fastapi uvicorn torch numpy requests

# Start server
python tts_service.py
```

### 3. Test the service
```bash
# Health check
curl http://localhost:8880/health

# Configuration check
curl http://localhost:8880/config

# Single TTS
curl -X POST "http://localhost:8880/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_heart"}' \
  --output test.wav
```

## API Endpoints

### Core TTS Endpoints

#### `POST /tts` - Single TTS
Convert single text to speech.

**Request:**
```json
{
  "text": "Hello world!",
  "voice": "af_heart",
  "language": "en-US",
  "speed": 1.0,
  "format": "wav"
}
```

#### `POST /tts/batch` - Batch TTS
Convert multiple texts to speech, returns ZIP archive.

**Request:**
```json
{
  "requests": [
    {"text": "Hello", "voice": "af_heart"},
    {"text": "World", "voice": "am_adam"}
  ]
}
```

#### `POST /tts/stream` - Streaming TTS
Stream audio as it's generated with smart chunking.

**Request:**
```json
{
  "text": "Long text with [pause:1.0s] pause support.",
  "voice": "af_heart",
  "language": "en-US",
  "speed": 1.0,
  "format": "wav"
}
```

### Utility Endpoints

#### `GET /health` - Health Check
Returns service status and G2P availability.

#### `GET /config` - Configuration
Returns current service configuration.

#### `GET /voices` - Available Voices
Lists supported voice names.

#### `POST /test/smart-split` - Test Text Chunking
Test smart_split functionality.

#### `POST /test/g2p` - Test G2P Service
Test G2P service conversion.

## Deployment Examples

### Docker Compose

```yaml
version: '3.8'
services:
  tts-service:
    build: .
    ports:
      - "8880:8880"
    environment:
      - G2P_SERVICE_URL=http://g2p-service:5000
      - MAX_BATCH_SIZE=100
    depends_on:
      - g2p-service
  
  g2p-service:
    image: your-g2p-service:latest
    ports:
      - "5000:5000"
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tts-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tts-service
  template:
    metadata:
      labels:
        app: tts-service
    spec:
      containers:
      - name: tts-service
        image: tts-service:latest
        ports:
        - containerPort: 8880
        env:
        - name: G2P_SERVICE_URL
          value: "http://g2p-service:5000"
        - name: MAX_BATCH_SIZE
          value: "100"
```

### Production Configuration

For production deployments:

```bash
# High-throughput configuration
export MAX_BATCH_SIZE=100
export MAX_TOKENS_PER_CHUNK=800
export G2P_TIMEOUT=60

# Load balancer configuration
export HOST=0.0.0.0
export PORT=8880

# Remote G2P service with authentication
export G2P_SERVICE_URL=https://g2p-api.example.com
export G2P_TIMEOUT=30
```

## Performance Tuning

### Chunk Size Optimization

- **Small chunks** (100-200 tokens): Lower latency, more G2P requests
- **Large chunks** (400-800 tokens): Higher throughput, fewer requests
- **Balanced** (200-400 tokens): Good compromise for most use cases

### G2P Service Optimization

- Use connection pooling for high-throughput scenarios
- Consider caching phoneme conversions for repeated text
- Monitor G2P service response times and adjust timeout accordingly

### Batch Processing

- Increase `MAX_BATCH_SIZE` for bulk processing scenarios
- Consider async G2P requests for large batches
- Monitor memory usage with large batch sizes

## Troubleshooting

### Common Issues

1. **G2P Service Unavailable**
   - Check `G2P_SERVICE_URL` configuration
   - Verify G2P service is running and accessible
   - Check network connectivity and firewall rules

2. **Timeout Errors**
   - Increase `G2P_TIMEOUT` for slow G2P services
   - Check G2P service performance and scaling

3. **Memory Issues**
   - Reduce `MAX_BATCH_SIZE` for memory-constrained environments
   - Reduce `MAX_TOKENS_PER_CHUNK` for large texts

4. **Audio Quality Issues**
   - Verify voice files are properly loaded
   - Check phoneme conversion accuracy
   - Test with different chunk sizes

### Monitoring

Monitor these metrics for production deployments:

- G2P service response times
- Audio generation latency
- Memory usage during batch processing
- Error rates for different endpoints
- Client connection patterns for streaming

## License

This service maintains license compliance by isolating GPL dependencies in the external G2P service while keeping the main TTS service GPL-free.