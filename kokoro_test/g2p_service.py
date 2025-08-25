#!/usr/bin/env python3
"""
Minimal G2P Service - GPL Isolated

This service isolates GPL-licensed dependencies (misaki + espeak-ng)
from production code by running as a separate HTTP service.

Usage:
    python g2p_service.py

API:
    POST /convert
    Input: {"text": "hello world", "lang": "en"}
    Output: {"phonemes": "həˈloʊ wɜrld"}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from misaki import en, espeak
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="G2P Service",
    description="GPL-isolated Grapheme-to-Phoneme conversion service",
    version="1.0.0"
)

# Initialize G2P processors
try:
    fallback = espeak.EspeakFallback(british=False)
except Exception as e:
    logger.warning(f"EspeakFallback not enabled: {e}")
    fallback = None

g2p_en = en.G2P(trf=False, british=False, fallback=fallback, unk='')

# Request/Response models
class G2PRequest(BaseModel):
    text: str
    lang: str = "en"

class G2PResponse(BaseModel):
    phonemes: str
    text: str
    lang: str

class HealthResponse(BaseModel):
    status: str
    service: str

@app.post("/convert", response_model=G2PResponse)
async def convert_text_to_phonemes(request: G2PRequest):
    """Convert text to phonemes using misaki G2P."""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text field cannot be empty")
        
        if request.lang not in ['en', 'en-us']:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {request.lang}")
        
        # Convert text to phonemes
        _, tokens = g2p_en(request.text)
        phonemes = ''.join((t.phonemes or '') + (' ' if t.whitespace else '') for t in tokens).strip()
        
        return G2PResponse(
            phonemes=phonemes,
            text=request.text,
            lang=request.lang
        )
        
    except Exception as e:
        logger.error(f"G2P conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="g2p")

if __name__ == '__main__':
    print("Starting G2P Service (GPL Licensed)")
    print("This service isolates GPL dependencies from production code")
    print("API: POST /convert with {'text': 'hello', 'lang': 'en'}")
    print("Docs available at: http://localhost:5000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")