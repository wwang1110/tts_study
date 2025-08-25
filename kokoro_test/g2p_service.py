#!/usr/bin/env python3
"""
Multi-Language G2P Service - GPL Isolated

This service isolates GPL-licensed dependencies (misaki + espeak-ng) 
from production code by running as a separate HTTP service.

Supported Languages (based on kokoro/pipeline.py):
🇺🇸 'a' => American English, 🇬🇧 'b' => British English
🇪🇸 'e' => Spanish, 🇫🇷 'f' => French, 🇧🇷 'p' => Brazilian Portuguese
🇮🇳 'h' => Hindi, 🇮🇹 'i' => Italian
🇯🇵 'j' => Japanese (requires misaki[ja])
🇨🇳 'z' => Mandarin Chinese (requires misaki[zh])

Usage:
    python g2p_service.py

API:
    POST /convert
    Input: {"text": "hello world", "lang": "en"}
    Output: {"phonemes": "həˈloʊ wɜrld", "lang_name": "American English"}
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
    title="Multi-Language G2P Service",
    description="GPL-isolated Grapheme-to-Phoneme conversion service with multi-language support",
    version="2.0.0"
)

# Language code mappings (exact match to kokoro/pipeline.py)
ALIASES = {
    'en-us': 'a',
    'en-gb': 'b',
    'es': 'e',
    'fr-fr': 'f',
    'hi': 'h',
    'it': 'i',
    'pt-br': 'p',
    'ja': 'j',
    'zh': 'z',
}

LANG_CODES = {
    # pip install misaki[en]
    'a': 'American English',
    'b': 'British English',

    # espeak-ng
    'e': 'es',
    'f': 'fr-fr',
    'h': 'hi',
    'i': 'it',
    'p': 'pt-br',

    # pip install misaki[ja]
    'j': 'Japanese',

    # pip install misaki[zh]
    'z': 'Mandarin Chinese',
}

# Initialize G2P processors
g2p_processors = {}

# Initialize English G2P (American)
try:
    fallback_us = espeak.EspeakFallback(british=False)
except Exception as e:
    logger.warning(f"EspeakFallback (US) not enabled: {e}")
    fallback_us = None

g2p_processors['a'] = en.G2P(trf=False, british=False, fallback=fallback_us, unk='')

# Initialize English G2P (British)
try:
    fallback_gb = espeak.EspeakFallback(british=True)
except Exception as e:
    logger.warning(f"EspeakFallback (GB) not enabled: {e}")
    fallback_gb = None

g2p_processors['b'] = en.G2P(trf=False, british=True, fallback=fallback_gb, unk='')

# Initialize espeak-ng language processors
espeak_languages = {
    'e': 'es',           # Spanish
    'f': 'fr-fr',        # French
    'h': 'hi',           # Hindi
    'i': 'it',           # Italian
    'p': 'pt-br',        # Brazilian Portuguese
}

for code, espeak_lang in espeak_languages.items():
    try:
        g2p_processors[code] = espeak.EspeakG2P(language=espeak_lang)
        logger.info(f"Initialized {espeak_lang} G2P processor")
    except Exception as e:
        logger.warning(f"Failed to initialize {espeak_lang} G2P: {e}")

# Advanced languages (Japanese and Chinese) - optional
try:
    from misaki import ja
    g2p_processors['j'] = ja.JAG2P()
    logger.info("Initialized Japanese G2P processor")
except ImportError:
    logger.warning("Japanese G2P not available. Install with: pip install misaki[ja]")

try:
    from misaki import zh
    g2p_processors['z'] = zh.ZHG2P(version=None)
    logger.info("Initialized Chinese G2P processor")
except ImportError:
    logger.warning("Chinese G2P not available. Install with: pip install misaki[zh]")

logger.info(f"Total G2P processors initialized: {len(g2p_processors)}")
logger.info(f"Available languages: {list(g2p_processors.keys())}")

# Request/Response models
class G2PRequest(BaseModel):
    text: str
    lang: str = "en"

class G2PResponse(BaseModel):
    phonemes: str
    text: str
    lang: str
    lang_name: str

class HealthResponse(BaseModel):
    status: str
    service: str
    supported_languages: dict

@app.post("/convert", response_model=G2PResponse)
async def convert_text_to_phonemes(request: G2PRequest):
    """Convert text to phonemes using appropriate G2P processor."""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text field cannot be empty")
        
        # Normalize and resolve language code using aliases
        lang_code = request.lang.lower()
        lang_code = ALIASES.get(lang_code, lang_code)
        
        if lang_code not in g2p_processors:
            supported = list(g2p_processors.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported language: {request.lang} -> {lang_code}. Supported: {supported}"
            )
        
        # Get appropriate G2P processor
        g2p = g2p_processors[lang_code]
        
        # Convert text to phonemes based on processor type
        if lang_code in ['a', 'b']:
            # English variants return (graphemes, tokens)
            _, tokens = g2p(request.text)
            phonemes = ''.join((t.phonemes or '') + (' ' if t.whitespace else '') for t in tokens).strip()
        elif lang_code == 'j':
            # Japanese returns (phonemes, tokens)
            phonemes, _ = g2p(request.text)
        elif lang_code == 'z':
            # Chinese returns (phonemes, tokens)
            phonemes, _ = g2p(request.text)
        else:
            # espeak-ng languages (e, f, h, i, p) return (phonemes, graphemes)
            phonemes, _ = g2p(request.text)
        
        logger.info(f"Converted '{request.text}' ({request.lang} -> {lang_code}) -> '{phonemes}'")
        
        return G2PResponse(
            phonemes=phonemes,
            text=request.text,
            lang=lang_code,
            lang_name=LANG_CODES[lang_code]
        )
        
    except Exception as e:
        logger.error(f"G2P conversion error for {request.lang}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with supported languages."""
    return HealthResponse(
        status="healthy", 
        service="g2p",
        supported_languages=LANG_CODES
    )

@app.get("/languages")
async def list_languages():
    """List all supported languages."""
    return {
        "supported_languages": LANG_CODES,
        "language_codes": list(g2p_processors.keys()),
        "aliases": ALIASES
    }

if __name__ == '__main__':
    print("Starting Multi-Language G2P Service (GPL Licensed)")
    print("This service isolates GPL dependencies from production code")
    print("\nSupported Languages:")
    
    # Show only available processors
    available_langs = {code: LANG_CODES[code] for code in g2p_processors.keys() if code in LANG_CODES}
    for code, name in available_langs.items():
        emoji_map = {
            "American English": "🇺🇸", "British English": "🇬🇧",
            "Spanish": "🇪🇸", "French": "🇫🇷", "Brazilian Portuguese": "🇧🇷",
            "Hindi": "🇮🇳", "Italian": "🇮🇹", "Japanese": "🇯🇵", "Mandarin Chinese": "🇨🇳"
        }
        emoji = emoji_map.get(name, "🌍")
        print(f"  {emoji} '{code}' => {name}")
    
    print(f"\nLanguage Aliases:")
    for alias, code in ALIASES.items():
        if code in g2p_processors:
            print(f"  '{alias}' -> '{code}' ({LANG_CODES[code]})")
    
    print(f"\nAPI Endpoints:")
    print(f"  POST /convert - Convert text to phonemes")
    print(f"  GET /health - Service health check")
    print(f"  GET /languages - List supported languages")
    print(f"  GET /docs - API documentation")
    print(f"\nExample: POST /convert with {{'text': 'hello', 'lang': 'en'}}")
    print(f"Server starting at: http://localhost:5000")
    
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")