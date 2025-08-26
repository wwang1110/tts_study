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
from typing import List, Optional, AsyncGenerator, Tuple
import wave
import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

# Import our license-safe pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from safe_pipeline import SafePipeline

# Initialize FastAPI app
app = FastAPI(
    title="License-Safe Kokoro TTS Service",
    description="TTS service using GPL-isolated G2P service",
    version="1.0.0"
)

# Global pipeline instance
pipeline = None

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

# Simple smart_split implementation
async def simple_smart_split(
    text: str,
    max_tokens: int = 400,
    min_tokens: int = 100
) -> AsyncGenerator[str, None]:
    """
    Simple version of smart_split that breaks text into chunks.
    
    Args:
        text: Input text to split
        max_tokens: Maximum tokens per chunk (approximated by character count / 4)
        min_tokens: Minimum tokens per chunk
        
    Yields:
        Text chunks
    """
    # Simple approximation: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    min_chars = min_tokens * 4
    
    # Handle pause tags first
    pause_pattern = re.compile(r'\[pause:(\d+(?:\.\d+)?)s\]', re.IGNORECASE)
    parts = pause_pattern.split(text)
    
    for i, part in enumerate(parts):
        if i % 2 == 1:  # This is a pause duration
            # Yield pause as special marker
            yield f"__PAUSE__{part}__"
            continue
            
        if not part.strip():
            continue
            
        # Split text part into sentences
        sentences = re.split(r'([.!?;:])\s*', part)
        
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
async def text_to_phonemes(text: str, language: str, g2p_url: str = "http://localhost:5000") -> str:
    """Convert text to phonemes using G2P service"""
    import requests
    
    try:
        response = requests.post(
            f"{g2p_url}/convert",
            json={"text": text, "lang": language},
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        if 'error' in data:
            raise RuntimeError(f"G2P service error: {data['error']}")
        
        return data['phonemes']
        
    except requests.RequestException as e:
        raise RuntimeError(f"G2P service unavailable: {e}")
    except ImportError:
        raise RuntimeError("requests library required for G2P service. Install with: pip install requests")

# Utility functions
def audio_to_wav_bytes(audio_tensor, sample_rate=24000):
    """Convert audio tensor to WAV bytes"""
    # Convert to numpy if it's a tensor
    if hasattr(audio_tensor, 'numpy'):
        audio_np = audio_tensor.numpy()
    else:
        audio_np = audio_tensor
    
    # Ensure int16 format
    if audio_np.dtype != np.int16:
        if audio_np.dtype == np.float32:
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_np.astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_np.tobytes())
    
    return wav_buffer.getvalue()

def audio_to_pcm_bytes(audio_tensor):
    """Convert audio tensor to raw PCM bytes"""
    if hasattr(audio_tensor, 'numpy'):
        audio_np = audio_tensor.numpy()
    else:
        audio_np = audio_tensor
    
    if audio_np.dtype != np.int16:
        if audio_np.dtype == np.float32:
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_np.astype(np.int16)
    
    return audio_np.tobytes()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the TTS pipeline"""
    global pipeline
    try:
        pipeline = SafePipeline()
        print("✅ SafePipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize SafePipeline: {e}")
        raise

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    g2p_available = True
    try:
        import requests
        response = requests.get("http://localhost:5000/health", timeout=5)
        g2p_available = response.status_code == 200
    except:
        g2p_available = False
    
    return {
        "status": "healthy" if pipeline else "unhealthy",
        "pipeline_ready": pipeline is not None,
        "g2p_service_available": g2p_available,
        "supported_languages": ["en-US", "en-GB", "es", "fr-FR", "pt-BR", "hi", "it", "ja", "zh"],
        "supported_formats": ["wav", "pcm"]
    }

# Mode 1: Single TTS
@app.post("/tts")
async def single_tts(request: TTSRequest):
    """Single text-to-speech conversion using G2P service and phonemes"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    try:
        # Convert text to phonemes using G2P service
        phonemes = await text_to_phonemes(request.text, request.language)
        
        # Generate audio from phonemes
        if pipeline:  # Type guard
            audio_tensor = pipeline.from_phonemes(
                phonemes=phonemes,
                voice=request.voice,
                speed=request.speed
            )
        else:
            raise HTTPException(status_code=503, detail="Pipeline not available")
        
        # Convert to requested format
        if request.format.lower() == "wav":
            audio_bytes = audio_to_wav_bytes(audio_tensor)
            media_type = "audio/wav"
        elif request.format.lower() == "pcm":
            audio_bytes = audio_to_pcm_bytes(audio_tensor)
            media_type = "audio/pcm"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
        
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.{request.format}",
                "X-Audio-Duration": str(len(audio_tensor) / 24000),  # Duration in seconds
                "X-Voice-Used": request.voice,
                "X-Phonemes-Used": phonemes[:100] + "..." if len(phonemes) > 100 else phonemes
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

# Mode 2: Batch TTS
@app.post("/tts/batch")
async def batch_tts(batch_request: BatchTTSRequest):
    """Batch text-to-speech conversion using G2P service and phonemes"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    if len(batch_request.requests) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 requests per batch")
    
    try:
        # Create ZIP archive in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, req in enumerate(batch_request.requests):
                try:
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
                    
                except Exception as e:
                    # Add error file for failed requests
                    error_content = f"Error generating audio: {str(e)}\nText: {req.text[:100]}..."
                    zip_file.writestr(f"error_{i+1:03d}.txt", error_content)
        
        zip_bytes = zip_buffer.getvalue()
        
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=tts_batch.zip",
                "X-Total-Requests": str(len(batch_request.requests))
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# Mode 3: Streaming TTS with smart_split and G2P
@app.post("/tts/stream")
async def streaming_tts(request: StreamingTTSRequest, client_request: Request):
    """Streaming text-to-speech conversion using G2P service and phonemes"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    async def generate_audio_stream():
        try:
            chunk_count = 0
            
            # Use smart_split to break text into optimal chunks
            async for text_chunk in simple_smart_split(request.text):
                # Check if client disconnected
                if await client_request.is_disconnected():
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
                        print("Pipeline not available for chunk processing")
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
                        
                        # Small delay for realistic streaming
                        await asyncio.sleep(0.01)
                    
                    chunk_count += 1
                    print(f"Streamed chunk {chunk_count}: '{text_chunk[:50]}...' -> {len(phonemes)} phonemes")
                    
                except Exception as e:
                    # Log error but continue with next chunk
                    print(f"Error processing chunk {chunk_count}: {e}")
                    continue
                
        except Exception as e:
            # Send error as final chunk
            error_msg = f"Streaming error: {str(e)}"
            yield error_msg.encode()
    
    media_type = "audio/wav" if request.format.lower() == "wav" else "audio/pcm"
    
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
    # Basic voice list - in production this would query the actual voice files
    voices = [
        "af_heart", "af_bella", "af_sarah", "af_nicole", "af_sky",
        "am_adam", "am_michael", "am_eric", "am_liam", "am_onyx"
    ]
    return {"voices": voices}

# Test endpoint for smart_split functionality
@app.post("/test/smart-split")
async def test_smart_split(request: dict):
    """Test the smart_split functionality"""
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    chunks = []
    async for chunk in simple_smart_split(text):
        chunks.append(chunk)
    
    return {
        "original_text": text,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "total_chars": len(text),
        "avg_chunk_size": len(text) / len(chunks) if chunks else 0
    }

# Test endpoint for G2P conversion
@app.post("/test/g2p")
async def test_g2p(request: dict):
    """Test G2P service conversion"""
    text = request.get("text", "")
    language = request.get("language", "en-US")
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        phonemes = await text_to_phonemes(text, language)
        return {
            "text": text,
            "language": language,
            "phonemes": phonemes,
            "phoneme_length": len(phonemes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"G2P conversion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880)