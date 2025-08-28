# Kokoro G2P Service - GPL Isolated

This directory contains a GPL-isolated G2P (Grapheme-to-Phoneme) service for Kokoro TTS that completely separates GPL-licensed dependencies from production code.

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and start the service
docker-compose up --build -d

# Check service health
curl http://localhost:5000/health

# Test the service
curl -X POST http://localhost:5000/convert \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "lang": "en-US"}'
```

### Using Docker

```bash
# Build the image
docker build -t kokoro-g2p-service:latest .

# Run the container
docker run -d -p 5000:5000 --name kokoro-g2p-service kokoro-g2p-service:latest

# Check logs
docker logs kokoro-g2p-service
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python g2p_service.py
```

## 🌍 Supported Languages

| Code | Language | Engine | Example |
|------|----------|--------|---------|
| `en-US`, `a` | American English | misaki[en] | "Hello world" |
| `en-GB`, `b` | British English | misaki[en] | "Hello world" |
| `es`, `e` | Spanish | espeak-ng | "Hola mundo" |
| `fr-FR`, `f` | French | espeak-ng | "Bonjour le monde" |
| `pt-BR`, `p` | Brazilian Portuguese | espeak-ng | "Olá mundo" |
| `hi`, `h` | Hindi | espeak-ng | "नमस्ते दुनिया" |
| `it`, `i` | Italian | espeak-ng | "Ciao mondo" |
| `ja`, `j` | Japanese | misaki[ja] | "こんにちは世界" |
| `zh`, `z` | Mandarin Chinese | misaki[zh] | "你好世界" |

## 📡 API Endpoints

### Convert Text to Phonemes
```http
POST /convert
Content-Type: application/json

{
  "text": "Hello world",
  "lang": "en-US"
}
```

**Response:**
```json
{
  "phonemes": "həˈloʊ wɜrld",
  "text": "Hello world",
  "lang": "a",
  "lang_name": "American English"
}
```

### Health Check
```http
GET /health
```

### List Languages
```http
GET /languages
```

### API Documentation
```http
GET /docs
```

## 🔒 License Compliance

### GPL Isolation
- **GPL Components**: `misaki`, `espeak-ng` (contained in Docker container)
- **Production Code**: Uses HTTP API calls (no GPL dependencies)
- **Commercial Safe**: Production applications can use this service commercially

### Architecture Benefits
- ✅ Complete GPL isolation in containerized service
- ✅ Production code remains GPL-free
- ✅ Scalable multi-language support
- ✅ Production-ready with Gunicorn + 4 workers

## 🛠️ Production Deployment

### Environment Variables
```bash
# Optional: Configure workers
GUNICORN_WORKERS=4
GUNICORN_BIND=0.0.0.0:5000
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kokoro-g2p-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kokoro-g2p-service
  template:
    metadata:
      labels:
        app: kokoro-g2p-service
    spec:
      containers:
      - name: g2p-service
        image: kokoro-g2p-service:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 🧪 Testing

```bash
# Test multi-language support
python safe_demo.py

# Test specific language
curl -X POST http://localhost:5000/convert \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour le monde", "lang": "fr-FR"}'
```

## 📁 Files

- `g2p_service.py` - Main G2P service (GPL isolated)
- `safe_pipeline.py` - License-safe TTS pipeline
- `g2p_client.py` - HTTP client for G2P service
- `safe_demo.py` - Multi-language demo
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container definition
- `docker-compose.yml` - Service orchestration

## 🤝 Integration

```python
from safe_pipeline import SafePipeline

# Initialize pipeline
pipeline = SafePipeline()

# Generate audio from text (uses G2P service)
audio = pipeline.from_text(
    text="Hello world", 
    voice="af_heart", 
    lang="en-US",
    g2p_url="http://localhost:5000"
)

# Generate audio from phonemes (no G2P needed)
audio = pipeline.from_phonemes("həˈloʊ wɜrld", voice="af_heart")
```

This architecture ensures complete GPL compliance while maintaining full multi-language TTS functionality.