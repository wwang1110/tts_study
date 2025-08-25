# Kokoro TTS - License-Safe Implementation

This directory contains a complete license-compliant solution for Kokoro TTS that isolates GPL dependencies from production code.

## 🚀 Quick Start

### 1. Start the G2P Service
```bash
cd kokoro_g2p_service
docker-compose up --build -d
```

### 2. Run the Demo
```bash
python safe_demo.py
```

## 📁 Directory Structure

```
kokoro_test/
├── kokoro_g2p_service/          # GPL-isolated G2P service (containerized)
│   ├── g2p_service.py           # FastAPI G2P service
│   ├── requirements.txt         # GPL dependencies
│   ├── Dockerfile               # Container definition
│   ├── docker-compose.yml       # Service orchestration
│   └── README.md                # G2P service documentation
├── safe_pipeline.py             # License-safe TTS pipeline
├── g2p_client.py                # HTTP client for G2P service
├── safe_demo.py                 # Multi-language demo
└── README.md                    # This file
```

## 🔒 License Compliance Architecture

### GPL-Isolated Components (Container)
- **G2P Service**: `kokoro_g2p_service/`
- **Dependencies**: misaki, espeak-ng (GPL 3.0)
- **Isolation**: Docker container with HTTP API

### Production-Safe Components (GPL-Free)
- **Pipeline**: `safe_pipeline.py` - Main TTS pipeline
- **Client**: `g2p_client.py` - HTTP client for G2P service
- **Demo**: `safe_demo.py` - Multi-language demonstration

## 🌍 Supported Languages

| Language | Code | Example Text |
|----------|------|--------------|
| American English | `en-US`, `a` | "Hello world" |
| British English | `en-GB`, `b` | "Hello world" |
| Spanish | `es`, `e` | "Hola mundo" |
| French | `fr-FR`, `f` | "Bonjour le monde" |
| Portuguese | `pt-BR`, `p` | "Olá mundo" |
| Hindi | `hi`, `h` | "नमस्ते दुनिया" |
| Italian | `it`, `i` | "Ciao mondo" |

## 🛠️ Usage Examples

### Direct Phonemes (100% GPL-Free)
```python
from safe_pipeline import SafePipeline

pipeline = SafePipeline()
audio = pipeline.from_phonemes("həˈloʊ wɜrld", voice="af_heart")
```

### Text with G2P Service (GPL Isolated)
```python
from safe_pipeline import SafePipeline

pipeline = SafePipeline()
audio = pipeline.from_text(
    text="Hello world", 
    voice="af_heart", 
    lang="en-US",
    g2p_url="http://localhost:5000"
)
```

### Multi-Language Support
```python
# Spanish
audio_es = pipeline.from_text("Hola mundo", voice="af_heart", lang="es")

# French  
audio_fr = pipeline.from_text("Bonjour le monde", voice="af_heart", lang="fr-FR")

# Portuguese
audio_pt = pipeline.from_text("Olá mundo", voice="af_heart", lang="pt-BR")
```

## 🚀 Deployment

### Development
```bash
# Start G2P service
cd kokoro_g2p_service
python g2p_service.py

# Run demo
cd ..
python safe_demo.py
```

### Production
```bash
# Deploy G2P service
cd kokoro_g2p_service
docker-compose up -d

# Use in production code
from safe_pipeline import SafePipeline
pipeline = SafePipeline()
```

### Kubernetes
```bash
# Build and deploy
cd kokoro_g2p_service
docker build -t kokoro-g2p-service:latest .
kubectl apply -f k8s-deployment.yaml
```

## ✅ License Benefits

- **Commercial Safe**: Production code has zero GPL dependencies
- **Scalable**: G2P service can run in multiple containers
- **Flexible**: Choose between direct phonemes or text input
- **Compliant**: Complete GPL isolation in containerized service
- **Multi-Language**: Support for 9+ languages with standard codes

## 🧪 Testing

```bash
# Test the complete solution
python safe_demo.py

# Test G2P service directly
curl -X POST http://localhost:5000/convert \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "lang": "en-US"}'
```

## 📚 Documentation

- **G2P Service**: See `kokoro_g2p_service/README.md` for detailed service documentation
- **API Reference**: Visit `http://localhost:5000/docs` when service is running
- **Architecture**: See `LICENSE_COMPLIANCE_ARCHITECTURE.md` in parent directory

This implementation ensures complete GPL compliance while maintaining full Kokoro TTS functionality across multiple languages.