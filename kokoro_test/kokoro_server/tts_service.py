#!/usr/bin/env python3
"""
Basic FastAPI TTS Service with SafePipeline
Implements three modes: single TTS, batch TTS, and streaming TTS with smart_split
"""

import asyncio
import base64
import io
import re
import zipfile
from contextlib import asynccontextmanager
from typing import List, Optional, AsyncGenerator, Tuple
import wave
import numpy as np
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our license-safe pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from safe_pipeline import SafePipeline

# Configuration
class Config:
    """Service configuration"""
    def __init__(self):
        self.g2p_url = os.getenv("G2P_SERVICE_URL", "http://0.0.0.0:5000")
        self.g2p_timeout = int(os.getenv("G2P_TIMEOUT", "30"))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "50"))
        self.max_tokens_per_chunk = int(os.getenv("MAX_TOKENS_PER_CHUNK", "400"))
        self.min_tokens_per_chunk = int(os.getenv("MIN_TOKENS_PER_CHUNK", "100"))
        # First chunk optimization for faster time to first token
        self.first_chunk_max_tokens = int(os.getenv("FIRST_CHUNK_MAX_TOKENS", "50"))
        self.first_chunk_min_tokens = int(os.getenv("FIRST_CHUNK_MIN_TOKENS", "25"))
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8880"))

# Global configuration and pipeline
config = Config()
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global pipeline
    try:
        logger.info("Initializing SafePipeline...")
        pipeline = SafePipeline()
        logger.info("✅ SafePipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize SafePipeline: {e}")
        raise
    
    yield
    
    # Shutdown (if needed)
    logger.info("TTS service shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="License-Safe Kokoro TTS Service",
    description="TTS service using GPL-isolated G2P service",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default="af_heart", description="Voice name")
    language: str = Field(default="en-US", description="Language code")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed")
    format: str = Field(default="wav", description="Audio format (wav/pcm)")

class BatchTTSRequest(BaseModel):
    requests: List[TTSRequest] = Field(..., description="List of TTS requests")

class StreamingTTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default="af_heart", description="Voice name")
    language: str = Field(default="en-US", description="Language code")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed")
    format: str = Field(default="wav", description="Audio format")

# Optimized smart_split implementation for faster time to first token
async def simple_smart_split(
    text: str,
    max_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None
) -> AsyncGenerator[str, None]:
    """
    Optimized version of smart_split that prioritizes fast time to first token.
    
    The first chunk is kept small (25-50 tokens, ideally one sentence) for faster
    initial audio generation, while subsequent chunks use normal sizing for quality.
    
    Args:
        text: Input text to split
        max_tokens: Maximum tokens per chunk for non-first chunks (uses config default if None)
        min_tokens: Minimum tokens per chunk for non-first chunks (uses config default if None)
        
    Yields:
        Text chunks (first chunk optimized for speed, rest for quality)
    """
    # Use config defaults if not specified
    if max_tokens is None:
        max_tokens = config.max_tokens_per_chunk
    if min_tokens is None:
        min_tokens = config.min_tokens_per_chunk
    
    # Simple approximation: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    min_chars = min_tokens * 4
    
    # First chunk optimization parameters
    first_chunk_max_chars = config.first_chunk_max_tokens * 4
    first_chunk_min_chars = config.first_chunk_min_tokens * 4
    
    # Handle pause tags first
    pause_pattern = re.compile(r'\[pause:(\d+(?:\.\d+)?)s\]', re.IGNORECASE)
    parts = pause_pattern.split(text)
    
    is_first_text_chunk = True
    
    for i, part in enumerate(parts):
        if i % 2 == 1:  # This is a pause duration
            # Yield pause as special marker
            yield f"__PAUSE__{part}__"
            continue
            
        if not part.strip():
            continue
            
        # Split text part into sentences
        sentences = re.split(r'([.!?;:])\s*', part)
        
        # Handle first chunk specially for faster time to first token
        if is_first_text_chunk and sentences:
            is_first_text_chunk = False
            
            # Try to get the first sentence
            first_sentence = sentences[0].strip() if sentences[0] else ""
            first_punct = sentences[1] if len(sentences) > 1 else ""
            
            if first_sentence:
                first_full_sentence = first_sentence + first_punct
                
                # Check if first sentence fits within first chunk limits
                if len(first_full_sentence) <= first_chunk_max_chars:
                    # Perfect! First sentence fits in first chunk
                    yield first_full_sentence.strip()
                    
                    # Continue with remaining sentences using normal logic
                    remaining_sentences = sentences[2:] if len(sentences) > 2 else []
                else:
                    # First sentence is too long, truncate at word boundary
                    words = first_sentence.split()
                    truncated_sentence = ""
                    
                    for word in words:
                        test_sentence = truncated_sentence + (" " if truncated_sentence else "") + word
                        if len(test_sentence) <= first_chunk_max_chars - len(first_punct):
                            truncated_sentence = test_sentence
                        else:
                            break
                    
                    if truncated_sentence:
                        # Yield truncated first chunk
                        yield (truncated_sentence + first_punct).strip()
                        
                        # Put remaining words back into sentences for normal processing
                        remaining_words = first_sentence[len(truncated_sentence):].strip()
                        if remaining_words:
                            remaining_sentences = [remaining_words + first_punct] + sentences[2:]
                        else:
                            remaining_sentences = sentences[2:] if len(sentences) > 2 else []
                    else:
                        # Edge case: even first word is too long, just yield it
                        yield first_full_sentence.strip()
                        remaining_sentences = sentences[2:] if len(sentences) > 2 else []
                
                # Process remaining sentences with normal chunking logic
                if remaining_sentences:
                    current_chunk = ""
                    for j in range(0, len(remaining_sentences), 2):
                        sentence = remaining_sentences[j].strip() if j < len(remaining_sentences) else ""
                        punct = remaining_sentences[j + 1] if j + 1 < len(remaining_sentences) else ""
                        
                        if not sentence:
                            continue
                            
                        full_sentence = sentence + punct
                        
                        # Check if adding this sentence exceeds max_chars
                        if len(current_chunk) + len(full_sentence) > max_chars and len(current_chunk) >= min_chars:
                            # Yield current chunk and start new one
                            if current_chunk.strip():
                                yield current_chunk.strip()
                            current_chunk = full_sentence
                        else:
                            # Add to current chunk
                            if current_chunk:
                                current_chunk += " " + full_sentence
                            else:
                                current_chunk = full_sentence
                    
                    # Don't forget the last chunk
                    if current_chunk.strip():
                        yield current_chunk.strip()
            continue
        
        # Normal processing for non-first chunks
        current_chunk = ""
        for j in range(0, len(sentences), 2):
            sentence = sentences[j].strip()
            punct = sentences[j + 1] if j + 1 < len(sentences) else ""
            
            if not sentence:
                continue
                
            full_sentence = sentence + punct
            
            # Check if adding this sentence exceeds max_chars
            if len(current_chunk) + len(full_sentence) > max_chars and len(current_chunk) >= min_chars:
                # Yield current chunk and start new one
                if current_chunk.strip():
                    yield current_chunk.strip()
                current_chunk = full_sentence
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + full_sentence
                else:
                    current_chunk = full_sentence
        
        # Don't forget the last chunk
        if current_chunk.strip():
            yield current_chunk.strip()

# G2P Service Integration
async def text_to_phonemes(text: str, language: str, g2p_url: Optional[str] = None) -> str:
    """Convert text to phonemes using configurable G2P service"""
    import requests
    
    # Use config default if not specified
    if g2p_url is None:
        g2p_url = config.g2p_url
    
    try:
        response = requests.post(
            f"{g2p_url}/convert",
            json={"text": text, "lang": language},
            timeout=config.g2p_timeout
        )
        response.raise_for_status()
        
        data = response.json()
        if 'error' in data:
            raise RuntimeError(f"G2P service error: {data['error']}")
        
        return data['phonemes']
        
    except requests.RequestException as e:
        raise RuntimeError(f"G2P service unavailable at {g2p_url}: {e}")
    except ImportError:
        raise RuntimeError("requests library required for G2P service. Install with: pip install requests")

# Utility functions
def audio_to_wav_bytes(audio_tensor, sample_rate=24000):
    """Convert audio tensor to WAV bytes"""
    logger.debug(f"Converting audio to WAV: sample_rate={sample_rate}")
    
    # Convert to numpy if it's a tensor
    if hasattr(audio_tensor, 'numpy'):
        audio_np = audio_tensor.numpy()
        logger.debug(f"Converted tensor to numpy: shape={audio_np.shape}, dtype={audio_np.dtype}")
    else:
        audio_np = audio_tensor
        logger.debug(f"Using numpy array: shape={audio_np.shape}, dtype={audio_np.dtype}")
    
    # Ensure int16 format
    original_dtype = audio_np.dtype
    if audio_np.dtype != np.int16:
        if audio_np.dtype == np.float32:
            audio_np = (audio_np * 32767).astype(np.int16)
            logger.debug(f"Converted float32 to int16: {original_dtype} -> {audio_np.dtype}")
        else:
            audio_np = audio_np.astype(np.int16)
            logger.debug(f"Converted {original_dtype} to int16")
    else:
        logger.debug("Audio already in int16 format")
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_np.tobytes())
    
    wav_bytes = wav_buffer.getvalue()
    duration_seconds = len(audio_np) / sample_rate
    logger.info(f"WAV created: {len(wav_bytes):,} bytes, duration={duration_seconds:.2f}s, samples={len(audio_np):,}")
    
    return wav_bytes

def audio_to_pcm_bytes(audio_tensor):
    """Convert audio tensor to raw PCM bytes"""
    logger.debug("Converting audio to PCM")
    
    if hasattr(audio_tensor, 'numpy'):
        audio_np = audio_tensor.numpy()
        logger.debug(f"Converted tensor to numpy: shape={audio_np.shape}, dtype={audio_np.dtype}")
    else:
        audio_np = audio_tensor
        logger.debug(f"Using numpy array: shape={audio_np.shape}, dtype={audio_np.dtype}")
    
    original_dtype = audio_np.dtype
    if audio_np.dtype != np.int16:
        if audio_np.dtype == np.float32:
            audio_np = (audio_np * 32767).astype(np.int16)
            logger.debug(f"Converted float32 to int16: {original_dtype} -> {audio_np.dtype}")
        else:
            audio_np = audio_np.astype(np.int16)
            logger.debug(f"Converted {original_dtype} to int16")
    else:
        logger.debug("Audio already in int16 format")
    
    pcm_bytes = audio_np.tobytes()
    duration_seconds = len(audio_np) / 24000  # Assuming 24kHz sample rate
    logger.info(f"PCM created: {len(pcm_bytes):,} bytes, duration={duration_seconds:.2f}s, samples={len(audio_np):,}")
    
    return pcm_bytes


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    
    g2p_available = True
    try:
        import requests
        response = requests.get(f"{config.g2p_url}/health", timeout=5)
        g2p_available = response.status_code == 200
        logger.debug(f"G2P service health check: {response.status_code}")
    except Exception as e:
        g2p_available = False
        logger.warning(f"G2P service health check failed: {e}")
    
    health_status = {
        "status": "healthy" if pipeline else "unhealthy",
        "pipeline_ready": pipeline is not None,
        "g2p_service_available": g2p_available,
        "g2p_service_url": config.g2p_url,
        "supported_languages": ["en-US", "en-GB", "es", "fr-FR", "pt-BR", "hi", "it", "ja", "zh"],
        "supported_formats": ["wav", "pcm"],
        "config": {
            "g2p_url": config.g2p_url,
            "g2p_timeout": config.g2p_timeout,
            "max_batch_size": config.max_batch_size,
            "max_tokens_per_chunk": config.max_tokens_per_chunk,
            "min_tokens_per_chunk": config.min_tokens_per_chunk,
            "first_chunk_max_tokens": config.first_chunk_max_tokens,
            "first_chunk_min_tokens": config.first_chunk_min_tokens
        }
    }
    
    logger.info(f"Health check response: status={health_status['status']}, pipeline_ready={health_status['pipeline_ready']}, g2p_available={g2p_available}")
    return health_status

# Mode 1: Single TTS
@app.post("/tts")
async def single_tts(request: TTSRequest):
    """Single text-to-speech conversion using G2P service and phonemes"""
    logger.info(f"Single TTS request: text_len={len(request.text)}, voice={request.voice}, lang={request.language}, format={request.format}")
    
    if not pipeline:
        logger.error("TTS pipeline not ready for single TTS request")
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    try:
        # Convert text to phonemes using G2P service
        logger.debug(f"Converting text to phonemes: '{request.text[:50]}...'")
        phonemes = await text_to_phonemes(request.text, request.language)
        logger.debug(f"G2P conversion successful: {len(phonemes)} phonemes")
        
        # Generate audio from phonemes
        logger.debug(f"Generating audio from phonemes using voice={request.voice}, speed={request.speed}")
        if pipeline:  # Type guard
            audio_tensor = pipeline.from_phonemes(
                phonemes=phonemes,
                voice=request.voice,
                speed=request.speed
            )
        else:
            logger.error("Pipeline not available during audio generation")
            raise HTTPException(status_code=503, detail="Pipeline not available")
        
        logger.debug(f"Audio generation successful: {len(audio_tensor)} samples")
        
        # Convert to requested format
        if request.format.lower() == "wav":
            audio_bytes = audio_to_wav_bytes(audio_tensor)
            media_type = "audio/wav"
        elif request.format.lower() == "pcm":
            audio_bytes = audio_to_pcm_bytes(audio_tensor)
            media_type = "audio/pcm"
        else:
            logger.error(f"Unsupported audio format requested: {request.format}")
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
        
        duration = len(audio_tensor) / 24000
        logger.info(f"Single TTS completed: {len(audio_bytes)} bytes, duration={duration:.2f}s, voice={request.voice}")
        
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.{request.format}",
                "X-Audio-Duration": str(duration),
                "X-Voice-Used": request.voice,
                "X-Phonemes-Used": phonemes[:100] + "..." if len(phonemes) > 100 else phonemes
            }
        )
        
    except Exception as e:
        logger.error(f"Single TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

# Mode 2: Batch TTS
@app.post("/tts/batch")
async def batch_tts(batch_request: BatchTTSRequest):
    """Batch text-to-speech conversion using G2P service and phonemes"""
    logger.info(f"Batch TTS request: {len(batch_request.requests)} requests")
    
    if not pipeline:
        logger.error("TTS pipeline not ready for batch TTS request")
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    if len(batch_request.requests) > config.max_batch_size:
        logger.error(f"Batch size {len(batch_request.requests)} exceeds maximum {config.max_batch_size}")
        raise HTTPException(status_code=400, detail=f"Maximum {config.max_batch_size} requests per batch")
    
    try:
        # Create ZIP archive in memory
        zip_buffer = io.BytesIO()
        successful_requests = 0
        failed_requests = 0
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, req in enumerate(batch_request.requests):
                try:
                    logger.debug(f"Processing batch item {i+1}/{len(batch_request.requests)}: voice={req.voice}, text_len={len(req.text)}")
                    
                    # Convert text to phonemes using G2P service
                    phonemes = await text_to_phonemes(req.text, req.language)
                    
                    # Generate audio from phonemes
                    if pipeline:  # Type guard
                        audio_tensor = pipeline.from_phonemes(
                            phonemes=phonemes,
                            voice=req.voice,
                            speed=req.speed
                        )
                    else:
                        raise Exception("Pipeline not available")
                    
                    # Convert to bytes
                    if req.format.lower() == "wav":
                        audio_bytes = audio_to_wav_bytes(audio_tensor)
                    else:
                        audio_bytes = audio_to_pcm_bytes(audio_tensor)
                    
                    # Add to ZIP
                    filename = f"tts_{i+1:03d}_{req.voice}.{req.format}"
                    zip_file.writestr(filename, audio_bytes)
                    
                    # Also add phonemes file for reference
                    phonemes_filename = f"phonemes_{i+1:03d}.txt"
                    zip_file.writestr(phonemes_filename, f"Text: {req.text}\nPhonemes: {phonemes}")
                    
                    successful_requests += 1
                    logger.debug(f"Batch item {i+1} completed successfully")
                    
                except Exception as e:
                    # Add error file for failed requests
                    error_content = f"Error generating audio: {str(e)}\nText: {req.text[:100]}..."
                    zip_file.writestr(f"error_{i+1:03d}.txt", error_content)
                    failed_requests += 1
                    logger.error(f"Batch item {i+1} failed: {str(e)}")
        
        zip_bytes = zip_buffer.getvalue()
        
        logger.info(f"Batch TTS completed: {successful_requests} successful, {failed_requests} failed, zip_size={len(zip_bytes)} bytes")
        
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=tts_batch.zip",
                "X-Total-Requests": str(len(batch_request.requests)),
                "X-Successful-Requests": str(successful_requests),
                "X-Failed-Requests": str(failed_requests)
            }
        )
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# Mode 3: Streaming TTS with smart_split and G2P
@app.post("/tts/stream")
async def streaming_tts(request: StreamingTTSRequest, client_request: Request):
    """Streaming text-to-speech conversion using G2P service and phonemes"""
    logger.info(f"Streaming TTS request: text_len={len(request.text)}, voice={request.voice}, lang={request.language}, format={request.format}")
    
    if not pipeline:
        logger.error("TTS pipeline not ready for streaming TTS request")
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    async def generate_audio_stream():
        chunk_count = 0
        total_bytes_streamed = 0
        
        try:
            logger.debug(f"Starting streaming generation for text: '{request.text[:100]}...'")
            
            # Use smart_split to break text into optimal chunks
            async for text_chunk in simple_smart_split(request.text):
                # Check if client disconnected
                if await client_request.is_disconnected():
                    logger.info(f"Client disconnected during streaming after {chunk_count} chunks")
                    break
                
                # Handle pause chunks
                if text_chunk.startswith("__PAUSE__") and text_chunk.endswith("__"):
                    pause_duration = float(text_chunk[9:-2])  # Extract duration
                    # Generate silence for pause duration
                    silence_samples = int(pause_duration * 24000)  # 24kHz sample rate
                    silence_audio = np.zeros(silence_samples, dtype=np.int16)
                    
                    if request.format.lower() == "wav":
                        silence_bytes = audio_to_wav_bytes(silence_audio)
                    else:
                        silence_bytes = audio_to_pcm_bytes(silence_audio)
                    
                    # Stream silence in chunks
                    chunk_size = 1024
                    for i in range(0, len(silence_bytes), chunk_size):
                        if await client_request.is_disconnected():
                            return
                        yield silence_bytes[i:i + chunk_size]
                        await asyncio.sleep(0.01)
                    
                    continue
                
                # Generate audio for text chunk using G2P + phonemes
                try:
                    # Convert text chunk to phonemes
                    phonemes = await text_to_phonemes(text_chunk, request.language)
                    
                    # Generate audio from phonemes
                    if pipeline:  # Type guard
                        audio_tensor = pipeline.from_phonemes(
                            phonemes=phonemes,
                            voice=request.voice,
                            speed=request.speed
                        )
                    else:
                        logger.error("Pipeline not available for chunk processing")
                        continue
                    
                    # Convert to bytes
                    if request.format.lower() == "wav":
                        audio_bytes = audio_to_wav_bytes(audio_tensor)
                    else:
                        audio_bytes = audio_to_pcm_bytes(audio_tensor)
                    
                    # Stream audio in chunks
                    chunk_size = 1024
                    for i in range(0, len(audio_bytes), chunk_size):
                        if await client_request.is_disconnected():
                            return
                        
                        chunk = audio_bytes[i:i + chunk_size]
                        yield chunk
                        total_bytes_streamed += len(chunk)
                        
                        # Small delay for realistic streaming
                        await asyncio.sleep(0.01)
                    
                    chunk_count += 1
                    logger.info(f"Streamed chunk {chunk_count}: '{text_chunk[:50]}...' -> {len(phonemes)} phonemes")
                    
                except Exception as e:
                    # Log error but continue with next chunk
                    logger.error(f"Error processing chunk {chunk_count}: {e}")
                    continue
                
        except Exception as e:
            # Send error as final chunk
            logger.error(f"Streaming generation error: {str(e)}")
            error_msg = f"Streaming error: {str(e)}"
            yield error_msg.encode()
        finally:
            logger.info(f"Streaming completed: {chunk_count} chunks processed, {total_bytes_streamed} total bytes streamed")
    
    media_type = "audio/wav" if request.format.lower() == "wav" else "audio/pcm"
    
    logger.debug(f"Starting streaming response with media_type={media_type}")
    
    return StreamingResponse(
        generate_audio_stream(),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=tts_stream.{request.format}",
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
            "X-Streaming-Mode": "g2p-phonemes-chunked"
        }
    )

# List available voices (basic implementation)
@app.get("/voices")
async def list_voices():
    """List available voices"""
    logger.info("Voices list requested")
    
    # Basic voice list - in production this would query the actual voice files
    voices = [
        "af_heart", "af_bella", "af_sarah", "af_nicole", "af_sky",
        "am_adam", "am_michael", "am_eric", "am_liam", "am_onyx"
    ]
    
    logger.debug(f"Returning {len(voices)} available voices")
    return {"voices": voices}

# Test endpoint for smart_split functionality
@app.post("/test/smart-split")
async def test_smart_split(request: dict):
    """Test the smart_split functionality"""
    text = request.get("text", "")
    logger.info(f"Smart split test requested: text_len={len(text)}")
    
    if not text:
        logger.error("Smart split test failed: no text provided")
        raise HTTPException(status_code=400, detail="Text is required")
    
    chunks = []
    async for chunk in simple_smart_split(text):
        chunks.append(chunk)
    
    result = {
        "original_text": text,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "total_chars": len(text),
        "avg_chunk_size": len(text) / len(chunks) if chunks else 0
    }
    
    logger.info(f"Smart split test completed: {len(chunks)} chunks, avg_size={result['avg_chunk_size']:.1f}")
    return result

# Test endpoint for G2P conversion
@app.post("/test/g2p")
async def test_g2p(request: dict):
    """Test G2P service conversion"""
    text = request.get("text", "")
    language = request.get("language", "en-US")
    
    logger.info(f"G2P test requested: text_len={len(text)}, language={language}")
    
    if not text:
        logger.error("G2P test failed: no text provided")
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        phonemes = await text_to_phonemes(text, language)
        result = {
            "text": text,
            "language": language,
            "phonemes": phonemes,
            "phoneme_length": len(phonemes)
        }
        
        logger.info(f"G2P test completed: {len(phonemes)} phonemes generated")
        return result
        
    except Exception as e:
        logger.error(f"G2P test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"G2P conversion failed: {str(e)}")

# Configuration endpoint
@app.get("/config")
async def get_config():
    """Get current service configuration"""
    logger.info("Configuration requested")
    
    config_data = {
        "g2p_url": config.g2p_url,
        "g2p_timeout": config.g2p_timeout,
        "max_batch_size": config.max_batch_size,
        "max_tokens_per_chunk": config.max_tokens_per_chunk,
        "min_tokens_per_chunk": config.min_tokens_per_chunk,
        "first_chunk_max_tokens": config.first_chunk_max_tokens,
        "first_chunk_min_tokens": config.first_chunk_min_tokens,
        "host": config.host,
        "port": config.port
    }
    
    logger.debug(f"Returning configuration: {config_data}")
    return config_data

if __name__ == "__main__":
    import uvicorn
    import socket
    
    logger.info("=" * 60)
    logger.info("🚀 License-Safe Kokoro TTS Service Starting")
    logger.info("=" * 60)
    
    # Check if port is available, if not, try alternative ports
    def find_available_port(start_port, max_attempts=10):
        logger.debug(f"Searching for available port starting from {start_port}")
        for i in range(max_attempts):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    logger.debug(f"Port {port} is available")
                    return port
            except OSError:
                logger.debug(f"Port {port} is in use")
                continue
        return None
    
    # Find available port
    available_port = find_available_port(config.port)
    if available_port is None:
        logger.error(f"Could not find available port starting from {config.port}")
        exit(1)
    
    if available_port != config.port:
        logger.warning(f"Port {config.port} is in use, using port {available_port} instead")
    
    logger.info(f"📡 G2P Service Endpoint: {config.g2p_url}")
    logger.info(f"📊 Batch Configuration: max_size={config.max_batch_size}")
    logger.info(f"🔧 Chunk Configuration: {config.min_tokens_per_chunk}-{config.max_tokens_per_chunk} tokens")
    logger.info(f"⚡ First Chunk Optimization: {config.first_chunk_min_tokens}-{config.first_chunk_max_tokens} tokens")
    logger.info(f"🌐 Server Address: http://{config.host}:{available_port}")
    logger.info(f"📚 API Documentation: http://{config.host}:{available_port}/docs")
    logger.info("=" * 60)
    
    logger.info("Starting uvicorn server...")
    
    # Configure uvicorn to use our logging
    uvicorn.run(
        app,
        host=config.host,
        port=available_port,
        log_level="info",
        access_log=True,
        use_colors=True
    )