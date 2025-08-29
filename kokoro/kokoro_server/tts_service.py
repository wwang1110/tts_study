#!/usr/bin/env python3
"""
Basic FastAPI TTS Service with SafePipeline
Implements three modes: single TTS, batch TTS, and streaming TTS with smart_split
"""

import asyncio
import base64
import io
import zipfile
from contextlib import asynccontextmanager
from typing import List, Optional, AsyncGenerator, Tuple
import numpy as np
import logging
import sys
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our license-safe pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kokoro import SafePipeline
from tts_components import (
    Config,
    TTSRequest,
    BatchTTSRequest,
    simple_smart_split,
    audio_to_wav_bytes,
    audio_to_pcm_bytes,
    create_wav_header,
    text_to_phonemes,
)

# Global configuration and pipeline
config = Config()

# Print configuration info
logger.info("=" * 60)
logger.info("📊 SafePipeline Configuration:")
logger.info(f"🔄 G2P Service URL: {config.g2p_url}")
logger.info(f"⏱️ G2P Timeout: {config.g2p_timeout} seconds")
logger.info(f"📦 Max Batch Size: {config.max_batch_size}")
logger.info(f"🧩 Max Tokens Per Chunk: {config.max_tokens_per_chunk}")
logger.info(f"⚡ First Chunk Max Tokens: {config.first_chunk_max_tokens}")
logger.info(f"🌐 Server Address: http://{config.host}:{config.port}")
logger.info(f"  DYNAMIC_BATCHING: {config.dynamic_batching}")
logger.info(f"  KOKORO_MAX_BATCH_SIZE: {config.kokoro_max_batch_size}")
logger.info(f"  NORMAL_QUEUE_MAX_WAIT_MS: {config.normal_queue_max_wait_ms}")
logger.info(f"  NORMAL_QUEUE_MIN_WAIT_MS: {config.normal_queue_min_wait_ms}")
logger.info(f"  HIGH_PRIORITY_QUEUE_MAX_WAIT_MS: {config.high_priority_queue_max_wait_ms}")
logger.info(f"  HIGH_PRIORITY_QUEUE_MIN_WAIT_MS: {config.high_priority_queue_min_wait_ms}")
logger.info(f"  MAX_QUEUE_SIZE: {config.max_queue_size}")
logger.info("=" * 60)

pipeline = None
# Global aiohttp session for connection pooling
g2p_session = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global pipeline, g2p_session
    try:
        logger.info("Initializing SafePipeline...")
        pipeline = SafePipeline(cache_dir="./.cache")
        
        # Initialize persistent aiohttp session for G2P service (if available)
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=config.g2p_timeout)
            g2p_session = aiohttp.ClientSession(timeout=timeout)
            logger.info("✅ G2P session pool initialized with aiohttp")
        except ImportError:
            logger.warning("⚠️ aiohttp not available, falling back to requests library")
            g2p_session = None
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    if g2p_session:
        await g2p_session.close()
        logger.info("G2P session pool closed")
    logger.info("TTS service shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="License-Safe Kokoro TTS Service",
    description="TTS service using GPL-isolated G2P service",
    version="1.0.0",
    lifespan=lifespan
)


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
            "first_chunk_max_tokens": config.first_chunk_max_tokens
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
        phonemes = await text_to_phonemes(
            request.text,
            request.language,
            g2p_session,
            config.g2p_url,
            config.g2p_timeout
        )
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
                    phonemes = await text_to_phonemes(
                        req.text,
                        req.language,
                        g2p_session,
                        config.g2p_url,
                        config.g2p_timeout
                    )
                    
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
async def streaming_tts(request: TTSRequest, client_request: Request):
    """Streaming text-to-speech conversion using G2P service and phonemes"""
    logger.info(f"Streaming TTS request: text_len={len(request.text)}, voice={request.voice}, lang={request.language}, format={request.format}")
    
    if not pipeline:
        logger.error("TTS pipeline not ready for streaming TTS request")
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    async def generate_audio_stream():
        chunk_count = 0
        total_bytes_streamed = 0
        wav_header_sent = False
        
        try:
            logger.debug(f"Starting TRUE streaming generation for text: '{request.text[:100]}...'")
            
            # Use smart_split to break text into optimal chunks
            async for text_chunk in simple_smart_split(
                request.text,
                config.max_tokens_per_chunk,
                config.first_chunk_max_tokens
            ):
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
                    
                    # Stream silence immediately
                    if request.format.lower() == "wav":
                        if not wav_header_sent:
                            # Send WAV header first (we'll update size later)
                            wav_header = create_wav_header(0)  # Placeholder size
                            yield wav_header
                            wav_header_sent = True
                        silence_bytes = silence_audio.tobytes()
                    else:
                        silence_bytes = silence_audio.tobytes()
                    
                    # Stream silence in chunks
                    chunk_size = 1024
                    for i in range(0, len(silence_bytes), chunk_size):
                        if await client_request.is_disconnected():
                            return
                        chunk = silence_bytes[i:i + chunk_size]
                        yield chunk
                        total_bytes_streamed += len(chunk)
                    continue
                
                # Generate audio for text chunk using G2P + phonemes
                try:
                    # Convert text chunk to phonemes
                    phonemes = await text_to_phonemes(
                        text_chunk,
                        request.language,
                        g2p_session,
                        config.g2p_url,
                        config.g2p_timeout
                    )
                    
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
                    
                    # Convert to numpy immediately
                    if hasattr(audio_tensor, 'numpy'):
                        audio_np = audio_tensor.numpy()
                    else:
                        audio_np = np.array(audio_tensor)
                    
                    # Ensure int16 format
                    if audio_np.dtype != np.int16:
                        if audio_np.dtype == np.float32:
                            audio_np = (audio_np * 32767).astype(np.int16)
                        else:
                            audio_np = audio_np.astype(np.int16)
                    
                    # Stream this chunk immediately
                    if request.format.lower() == "wav":
                        if not wav_header_sent:
                            # Send WAV header first (we'll update size later)
                            wav_header = create_wav_header(0)  # Placeholder size
                            yield wav_header
                            wav_header_sent = True
                        chunk_bytes = audio_np.tobytes()
                    else:
                        chunk_bytes = audio_np.tobytes()
                    
                    # Stream audio chunk in 1024-byte pieces
                    stream_chunk_size = 1024
                    for i in range(0, len(chunk_bytes), stream_chunk_size):
                        if await client_request.is_disconnected():
                            return
                        
                        stream_chunk = chunk_bytes[i:i + stream_chunk_size]
                        yield stream_chunk
                        total_bytes_streamed += len(stream_chunk)
                        
                        # Minimal delay for maximum streaming speed
                        await asyncio.sleep(0.0001)  # Further reduced for ultra-fast streaming
                    
                    chunk_count += 1
                    logger.info(f"Streamed chunk {chunk_count}: '{text_chunk[:50]}...' -> {len(audio_np)} samples")
                    
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
            logger.info(f"TRUE streaming completed: {chunk_count} chunks processed, {total_bytes_streamed} total bytes streamed")
    
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
    
    if pipeline and pipeline.voices:
        available_voices = sorted(list(pipeline.voices.keys()))
        logger.debug(f"Returning {len(available_voices)} cached voices")
        return {"voices": available_voices}

    # Fallback to querying the repo if pipeline not ready or no voices loaded
    try:
        from huggingface_hub import list_repo_files
        repo_files = list_repo_files(repo_id='hexgrad/Kokoro-82M', repo_type='model')
        voices = sorted([f.split('/')[-1].replace('.pt', '') for f in repo_files if f.startswith('voices/') and f.endswith('.pt')])
        logger.debug(f"Returning {len(voices)} voices from repo listing")
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Could not retrieve voice list from HuggingFace Hub: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve voice list")

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
    async for chunk in simple_smart_split(
        text,
        config.max_tokens_per_chunk,
        config.first_chunk_max_tokens
    ):
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
        phonemes = await text_to_phonemes(
            text,
            language,
            g2p_session,
            config.g2p_url,
            config.g2p_timeout
        )
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
        "first_chunk_max_tokens": config.first_chunk_max_tokens,
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