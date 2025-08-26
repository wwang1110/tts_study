#!/usr/bin/env python3
"""
Basic FastAPI TTS Service with SafePipeline
Implements three modes: single TTS, batch TTS, and streaming TTS
"""

import asyncio
import base64
import io
import zipfile
from typing import List, Optional
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
    """Single text-to-speech conversion"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    try:
        # Generate audio using safe pipeline
        audio_tensor = pipeline.from_text(
            text=request.text,
            voice=request.voice,
            lang=request.language,
            speed=request.speed
        )
        
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
                "X-Voice-Used": request.voice
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

# Mode 2: Batch TTS
@app.post("/tts/batch")
async def batch_tts(batch_request: BatchTTSRequest):
    """Batch text-to-speech conversion returning ZIP archive"""
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
                    # Generate audio
                    audio_tensor = pipeline.from_text(
                        text=req.text,
                        voice=req.voice,
                        lang=req.language,
                        speed=req.speed
                    )
                    
                    # Convert to bytes
                    if req.format.lower() == "wav":
                        audio_bytes = audio_to_wav_bytes(audio_tensor)
                    else:
                        audio_bytes = audio_to_pcm_bytes(audio_tensor)
                    
                    # Add to ZIP
                    filename = f"tts_{i+1:03d}_{req.voice}.{req.format}"
                    zip_file.writestr(filename, audio_bytes)
                    
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

# Mode 3: Streaming TTS
@app.post("/tts/stream")
async def streaming_tts(request: StreamingTTSRequest, client_request: Request):
    """Streaming text-to-speech conversion"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    async def generate_audio_stream():
        try:
            # Generate audio
            audio_tensor = pipeline.from_text(
                text=request.text,
                voice=request.voice,
                lang=request.language,
                speed=request.speed
            )
            
            # Convert to bytes
            if request.format.lower() == "wav":
                audio_bytes = audio_to_wav_bytes(audio_tensor)
            else:
                audio_bytes = audio_to_pcm_bytes(audio_tensor)
            
            # Stream in chunks
            chunk_size = 1024
            for i in range(0, len(audio_bytes), chunk_size):
                # Check if client disconnected
                if await client_request.is_disconnected():
                    break
                
                chunk = audio_bytes[i:i + chunk_size]
                yield chunk
                
                # Small delay to simulate real streaming
                await asyncio.sleep(0.01)
                
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
            "Transfer-Encoding": "chunked"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880)