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
import time

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
    StreamingTTSRequest,
    simple_smart_split,
    audio_to_wav_bytes,
    audio_to_pcm_bytes,
    create_wav_header,
    text_to_phonemes,
)
from tts_components.dynamic_batching import SingleProcessBatchQueue

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
logger.info(f"  KOKORO_MAX_WAIT_MS: {config.kokoro_max_wait_ms}")
logger.info(f"  KOKORO_MIN_WAIT_MS: {config.kokoro_min_wait_ms}")
logger.info(f"  MAX_QUEUE_SIZE: {config.max_queue_size}")
logger.info("=" * 60)
logger.info("✅ SafePipeline initialized successfully")

pipeline = None
# Global aiohttp session for connection pooling
g2p_session = None
batch_queue: Optional[SingleProcessBatchQueue] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global pipeline, g2p_session, batch_queue
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

        # Initialize batching if enabled
        if config.dynamic_batching:
            batch_queue = SingleProcessBatchQueue(
                max_batch_size=config.kokoro_max_batch_size,
                max_wait_ms=config.kokoro_max_wait_ms,
                min_wait_ms=config.kokoro_min_wait_ms,
                max_queue_size=config.max_queue_size
            )
            batch_queue.set_pipeline(pipeline)
            await batch_queue.start_worker()
            logger.info("✅ Dynamic batching enabled")
        else:
            logger.info("ℹ️ Dynamic batching disabled")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    if g2p_session:
        await g2p_session.close()
        logger.info("G2P session pool closed")
    if batch_queue:
        await batch_queue.stop_worker()
        logger.info("✅ Batch queue stopped")
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
    """Single text-to-speech conversion with dynamic batching"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    try:
        # Step 1: Convert text to phonemes (individual G2P call)
        phonemes = await text_to_phonemes(
            request.text,
            request.language,
            g2p_session,
            config.g2p_url,
            config.g2p_timeout
        )
        
        # Step 2: Submit to batch queue or process directly
        if batch_queue and config.dynamic_batching:
            # Use dynamic batching
            audio_tensor = await batch_queue.submit_for_batching(
                phonemes=phonemes,
                voice=request.voice,
                speed=request.speed
            )
        else:
            # Direct processing (fallback)
            audio_tensor = pipeline.from_phonemes(
                phonemes=phonemes,
                voice=request.voice,
                speed=request.speed
            )
        
        # Step 3: Convert to response format
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
                "X-Audio-Duration": str(len(audio_tensor) / 24000),
                "X-Voice-Used": request.voice,
                "X-Batching-Enabled": str(config.dynamic_batching)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

# Mode 2: Batch TTS
@app.post("/tts/batch")
async def batch_tts(batch_request: BatchTTSRequest):
    """Batch text-to-speech conversion with enhanced batching"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    if len(batch_request.requests) > config.max_batch_size:
        raise HTTPException(status_code=400, detail=f"Maximum {config.max_batch_size} requests per batch")
    
    try:
        # Step 1: Process all G2P calls concurrently
        g2p_tasks = []
        for req in batch_request.requests:
            task = text_to_phonemes(
                req.text,
                req.language,
                g2p_session,
                config.g2p_url,
                config.g2p_timeout
            )
            g2p_tasks.append(task)
        
        phonemes_list = await asyncio.gather(*g2p_tasks, return_exceptions=True)
        
        # Step 2: Submit all to batch queue or process directly
        if batch_queue and config.dynamic_batching:
            # Use dynamic batching
            kokoro_tasks = []
            valid_requests = []
            for i, (req, phonemes_result) in enumerate(zip(batch_request.requests, phonemes_list)):
                if isinstance(phonemes_result, Exception):
                    logger.error(f"G2P failed for batch item {i}: {phonemes_result}")
                    continue # Skip failed G2P requests
                
                valid_requests.append((req, phonemes_result))
                task = batch_queue.submit_for_batching(
                    phonemes=phonemes_result,
                    voice=req.voice,
                    speed=req.speed
                )
                kokoro_tasks.append(task)
            
            audio_results_list = await asyncio.gather(*kokoro_tasks, return_exceptions=True)
            
            # Create a dictionary to map original request to result
            results_dict = {}
            for (req, _), result in zip(valid_requests, audio_results_list):
                results_dict[req.text] = result

        else:
            # Direct processing
            results_dict = {}
            for req, phonemes_result in zip(batch_request.requests, phonemes_list):
                 if isinstance(phonemes_result, Exception):
                    results_dict[req.text] = phonemes_result
                    continue
                 audio = pipeline.from_phonemes(phonemes_result, req.voice, req.speed)
                 results_dict[req.text] = audio
        
        # Step 3: Create ZIP response
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, req in enumerate(batch_request.requests):
                audio_tensor = results_dict.get(req.text)
                try:
                    if isinstance(audio_tensor, Exception) or audio_tensor is None:
                        raise audio_tensor or Exception("Audio generation failed")

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
        
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=tts_batch.zip",
                "X-Total-Requests": str(len(batch_request.requests)),
                "X-Batching-Enabled": str(config.dynamic_batching)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# Mode 3: Streaming TTS with smart_split and G2P
@app.post("/tts/stream")
async def streaming_tts(request: StreamingTTSRequest, client_request: Request):
    """Streaming text-to-speech conversion with chunk-level batching"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    async def generate_audio_stream():
        try:
            chunk_count = 0
            
            # Use smart_split to break text into optimal chunks
            async for text_chunk in simple_smart_split(
                request.text,
                config.max_tokens_per_chunk,
                config.first_chunk_max_tokens
            ):
                # Check if client disconnected
                if await client_request.is_disconnected():
                    break
                
                # Handle pause chunks
                if text_chunk.startswith("__PAUSE__") and text_chunk.endswith("__"):
                    pause_duration = float(text_chunk[9:-2])
                    silence_samples = int(pause_duration * 24000)
                    silence_audio = np.zeros(silence_samples, dtype=np.int16)
                    
                    if request.format.lower() == "wav":
                        # This is tricky for streaming; for now, we send raw PCM for silence
                        silence_bytes = silence_audio.tobytes()
                    else:
                        silence_bytes = silence_audio.tobytes()
                    
                    # Stream silence in chunks
                    chunk_size = 1024
                    for i in range(0, len(silence_bytes), chunk_size):
                        if await client_request.is_disconnected():
                            return
                        yield silence_bytes[i:i + chunk_size]
                        await asyncio.sleep(0.01)
                    
                    continue
                
                # Generate audio for text chunk
                try:
                    # Convert text chunk to phonemes
                    phonemes = await text_to_phonemes(
                        text_chunk,
                        request.language,
                        g2p_session,
                        config.g2p_url,
                        config.g2p_timeout
                    )
                    
                    # Submit to batch queue or process directly
                    if batch_queue and config.dynamic_batching:
                        audio_tensor = await batch_queue.submit_for_batching(
                            phonemes=phonemes,
                            voice=request.voice,
                            speed=request.speed
                        )
                    else:
                        audio_tensor = pipeline.from_phonemes(
                            phonemes=phonemes,
                            voice=request.voice,
                            speed=request.speed
                        )
                    
                    # Convert to bytes
                    if request.format.lower() == "wav":
                        # For streaming WAV, we can't know the final size for the header.
                        # A common approach is to send raw PCM and let the client handle it,
                        # or use a library that can patch the header later.
                        # For simplicity, we'll stream raw PCM and the client can reconstruct the WAV.
                        audio_bytes = audio_to_pcm_bytes(audio_tensor)
                    else:
                        audio_bytes = audio_to_pcm_bytes(audio_tensor)
                    
                    # Stream audio in chunks
                    chunk_size = 1024
                    for i in range(0, len(audio_bytes), chunk_size):
                        if await client_request.is_disconnected():
                            return
                        
                        chunk = audio_bytes[i:i + chunk_size]
                        yield chunk
                        await asyncio.sleep(0.001) # smaller delay for faster streaming
                    
                    chunk_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_count}: {e}")
                    continue
                
        except Exception as e:
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
            "X-Batching-Enabled": str(config.dynamic_batching)
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
        "port": config.port,
        "dynamic_batching_enabled": config.dynamic_batching,
        "kokoro_max_batch_size": config.kokoro_max_batch_size,
        "kokoro_max_wait_ms": config.kokoro_max_wait_ms,
        "kokoro_min_wait_ms": config.kokoro_min_wait_ms,
    }
    
    logger.debug(f"Returning configuration: {config_data}")
    return config_data

@app.get("/health/batching")
async def batching_health():
    """Detailed health check for batching system"""
    if not batch_queue:
        return {"batching_enabled": False, "status": "disabled"}
    
    stats = batch_queue.get_stats()
    
    return {
        "batching_enabled": config.dynamic_batching,
        "status": "healthy" if batch_queue.running else "unhealthy",
        "queue_status": {
            "current_size": stats["current_queue_size"],
            "worker_running": stats["worker_running"]
        },
        "performance": {
            "total_batches": stats["total_batches"],
            "total_requests": stats["total_requests"],
            "avg_batch_size": stats["avg_batch_size"]
        },
        "config": {
            "max_batch_size": config.kokoro_max_batch_size,
            "max_wait_ms": config.kokoro_max_wait_ms,
            "min_wait_ms": config.kokoro_min_wait_ms
        }
    }

@app.post("/test/batching")
async def test_batching():
    """Test batching performance"""
    if not batch_queue:
        return {"error": "Batching not enabled"}
    
    # Send 4 concurrent requests to test batching
    tasks = []
    for i in range(4):
        task = batch_queue.submit_for_batching(
            phonemes=f"həˈloʊ wɜrld {i}",
            voice="af_heart",
            speed=1.0
        )
        tasks.append(task)
    
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = (time.time() - start_time) * 1000
    
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    
    return {
        "test_requests": 4,
        "successful": success_count,
        "total_time_ms": total_time,
        "avg_time_per_request_ms": total_time / 4,
        "batching_efficiency": "good" if total_time < 200 else "needs_tuning"
    }

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