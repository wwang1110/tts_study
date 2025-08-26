# Dynamic Batching Implementation Guide

## Overview

This document provides a step-by-step implementation guide for adding dynamic batching to the Kokoro TTS server using a single shared queue approach. The implementation focuses on batching only the Kokoro model inference while keeping G2P calls individual.

## Architecture Summary

```
REST Endpoint → G2P Conversion → Shared Batch Queue → Background Worker → Kokoro Model → Result via Future
```

## Step 1: Create Batch Entry Data Structure

Create a new file `kokoro_test/kokoro_server/batching.py`:

```python
#!/usr/bin/env python3
"""
Dynamic Batching System for Kokoro TTS Server
Single shared queue implementation with asyncio
"""

import asyncio
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class BatchEntry:
    """Entry for batching queue with result Future"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phonemes: str = ""
    voice: str = ""
    speed: float = 1.0
    result_future: asyncio.Future = field(default_factory=asyncio.Future)
    arrival_time: float = field(default_factory=time.time)
    
    def phoneme_length(self) -> int:
        return len(self.phonemes)
    
    def is_expired(self, timeout_seconds: float = 10.0) -> bool:
        return (time.time() - self.arrival_time) > timeout_seconds
```

## Step 2: Implement Single Process Batch Queue

Add to `batching.py`:

```python
class SingleProcessBatchQueue:
    """Single-process batching queue using asyncio primitives"""
    
    def __init__(self, 
                 max_batch_size: int = 4,
                 max_wait_ms: int = 50,
                 min_wait_ms: int = 10,
                 max_queue_size: int = 1000):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.min_wait_ms = min_wait_ms
        
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.worker_task: Optional[asyncio.Task] = None
        self.running = False
        self.pipeline = None  # Will be set during initialization
        
        # Thread pool for CPU-bound Kokoro inference
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Metrics
        self.total_batches = 0
        self.total_requests = 0
        self.batch_sizes = []
    
    def set_pipeline(self, pipeline):
        """Set the SafePipeline instance"""
        self.pipeline = pipeline
    
    async def start_worker(self):
        """Start the background batch worker"""
        if not self.running:
            self.running = True
            self.worker_task = asyncio.create_task(self._batch_worker())
            logger.info("Batch worker started")
    
    async def stop_worker(self):
        """Stop the background worker"""
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        logger.info("Batch worker stopped")
    
    async def submit_for_batching(self, phonemes: str, voice: str, speed: float = 1.0) -> torch.FloatTensor:
        """Submit request for batching and return result via Future"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        # Create entry with Future for result
        entry = BatchEntry(
            phonemes=phonemes,
            voice=voice,
            speed=speed
        )
        
        try:
            # Put in queue (non-blocking)
            self.queue.put_nowait(entry)
        except asyncio.QueueFull:
            raise RuntimeError("Batch queue is full - service overloaded")
        
        # Wait for result from background worker
        try:
            result = await asyncio.wait_for(entry.result_future, timeout=10.0)
            return result
        except asyncio.TimeoutError:
            raise RuntimeError("Batch processing timeout")
    
    async def _batch_worker(self):
        """Background worker that continuously processes batches"""
        logger.info("Batch worker loop started")
        
        while self.running:
            try:
                # Collect batch with adaptive timeout
                batch = await self._collect_batch()
                
                if batch:
                    # Process batch in background (don't block worker loop)
                    asyncio.create_task(self._process_batch(batch))
                    
            except Exception as e:
                logger.error(f"Batch worker error: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error
        
        logger.info("Batch worker loop ended")
    
    async def _collect_batch(self) -> List[BatchEntry]:
        """Collect entries for a batch with adaptive timeout"""
        batch = []
        
        # Wait for first request (blocking with timeout)
        try:
            first_entry = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            batch.append(first_entry)
        except asyncio.TimeoutError:
            return []  # No requests available
        
        # Calculate adaptive wait time based on current batch size
        wait_time_ms = self.max_wait_ms if len(batch) == 1 else self.min_wait_ms
        deadline = time.time() + (wait_time_ms / 1000.0)
        
        # Collect additional requests until batch is full or timeout
        while len(batch) < self.max_batch_size and time.time() < deadline:
            remaining_time = deadline - time.time()
            if remaining_time <= 0:
                break
            
            try:
                entry = await asyncio.wait_for(self.queue.get(), timeout=remaining_time)
                batch.append(entry)
            except asyncio.TimeoutError:
                break  # Timeout reached, process current batch
        
        return batch
    
    async def _process_batch(self, batch: List[BatchEntry]):
        """Process batch and set results on Futures"""
        batch_start_time = time.time()
        
        try:
            # Group by voice for efficient processing
            voice_groups = self._group_by_voice(batch)
            
            # Process each voice group
            for voice, entries in voice_groups.items():
                await self._process_voice_group(voice, entries)
            
            # Update metrics
            batch_time_ms = (time.time() - batch_start_time) * 1000
            self._update_metrics(len(batch), batch_time_ms)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exception on all Futures
            for entry in batch:
                if not entry.result_future.done():
                    entry.result_future.set_exception(e)
    
    def _group_by_voice(self, batch: List[BatchEntry]) -> Dict[str, List[BatchEntry]]:
        """Group batch entries by voice for efficient processing"""
        voice_groups = {}
        for entry in batch:
            if entry.voice not in voice_groups:
                voice_groups[entry.voice] = []
            voice_groups[entry.voice].append(entry)
        return voice_groups
    
    async def _process_voice_group(self, voice: str, entries: List[BatchEntry]):
        """Process entries with the same voice"""
        try:
            # Prepare data for batch processing
            phoneme_list = [entry.phonemes for entry in entries]
            speed = entries[0].speed  # Assume same speed for batch
            
            # Run Kokoro inference in thread pool (CPU-bound operation)
            loop = asyncio.get_event_loop()
            audio_results = await loop.run_in_executor(
                self.thread_pool,
                self._run_kokoro_batch,
                phoneme_list,
                voice,
                speed
            )
            
            # Set results on Futures (this unblocks the REST endpoints!)
            for entry, audio in zip(entries, audio_results):
                if not entry.result_future.done():
                    entry.result_future.set_result(audio)
                    
        except Exception as e:
            logger.error(f"Voice group processing failed for {voice}: {e}")
            # Set exception on all Futures in this group
            for entry in entries:
                if not entry.result_future.done():
                    entry.result_future.set_exception(e)
    
    def _run_kokoro_batch(self, phoneme_list: List[str], voice: str, speed: float) -> List[torch.FloatTensor]:
        """Run Kokoro inference for batch (CPU-bound, runs in thread pool)"""
        results = []
        
        for phonemes in phoneme_list:
            try:
                # Validate phoneme length (Kokoro limit)
                if len(phonemes) > 510:
                    logger.warning(f"Phonemes too long ({len(phonemes)}), truncating")
                    phonemes = phonemes[:510]
                
                # Generate audio using SafePipeline
                audio = self.pipeline.from_phonemes(phonemes, voice, speed)
                results.append(audio)
                
            except Exception as e:
                logger.error(f"Kokoro inference failed for phonemes: {e}")
                # Return silence as fallback (1 second at 24kHz)
                silence = torch.zeros(24000, dtype=torch.float32)
                results.append(silence)
        
        return results
    
    def _update_metrics(self, batch_size: int, batch_time_ms: float):
        """Update batch processing metrics"""
        self.total_batches += 1
        self.total_requests += batch_size
        self.batch_sizes.append(batch_size)
        
        # Keep only last 100 batch sizes for rolling average
        if len(self.batch_sizes) > 100:
            self.batch_sizes.pop(0)
        
        logger.debug(f"Processed batch: size={batch_size}, time={batch_time_ms:.1f}ms")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current batch processing statistics"""
        return {
            "total_batches": self.total_batches,
            "total_requests": self.total_requests,
            "avg_batch_size": sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0,
            "current_queue_size": self.queue.qsize(),
            "worker_running": self.running
        }
```

## Step 3: Update Configuration

Add to `kokoro_test/kokoro_server/.env.example`:

```bash
# Dynamic Batching Configuration
ENABLE_KOKORO_BATCHING=true
KOKORO_MAX_BATCH_SIZE=4
KOKORO_MAX_WAIT_MS=50
KOKORO_MIN_WAIT_MS=10
MAX_QUEUE_SIZE=1000
```

Update the `Config` class in `tts_service.py`:

```python
class Config:
    """Service configuration"""
    def __init__(self):
        # Existing config...
        self.g2p_url = os.getenv("G2P_SERVICE_URL", "http://localhost:5000")
        self.g2p_timeout = int(os.getenv("G2P_TIMEOUT", "30"))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "50"))
        self.max_tokens_per_chunk = int(os.getenv("MAX_TOKENS_PER_CHUNK", "400"))
        self.min_tokens_per_chunk = int(os.getenv("MIN_TOKENS_PER_CHUNK", "100"))
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8880"))
        
        # New batching config
        self.enable_kokoro_batching = os.getenv("ENABLE_KOKORO_BATCHING", "true").lower() == "true"
        self.kokoro_max_batch_size = int(os.getenv("KOKORO_MAX_BATCH_SIZE", "4"))
        self.kokoro_max_wait_ms = int(os.getenv("KOKORO_MAX_WAIT_MS", "50"))
        self.kokoro_min_wait_ms = int(os.getenv("KOKORO_MIN_WAIT_MS", "10"))
        self.max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", "1000"))
```

## Step 4: Modify TTS Service

Update `tts_service.py` to integrate batching:

```python
# Add imports at the top
from batching import SingleProcessBatchQueue

# Global batch queue
batch_queue: Optional[SingleProcessBatchQueue] = None

# Update startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the TTS pipeline with batching"""
    global pipeline, batch_queue, config
    
    try:
        # Initialize config
        config = Config()
        
        # Initialize pipeline
        pipeline = SafePipeline()
        print("✅ SafePipeline initialized successfully")
        
        # Initialize batching if enabled
        if config.enable_kokoro_batching:
            batch_queue = SingleProcessBatchQueue(
                max_batch_size=config.kokoro_max_batch_size,
                max_wait_ms=config.kokoro_max_wait_ms,
                min_wait_ms=config.kokoro_min_wait_ms,
                max_queue_size=config.max_queue_size
            )
            batch_queue.set_pipeline(pipeline)
            await batch_queue.start_worker()
            print("✅ Dynamic batching enabled")
        else:
            print("ℹ️ Dynamic batching disabled")
            
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        raise

# Update shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    global batch_queue
    
    if batch_queue:
        await batch_queue.stop_worker()
        print("✅ Batch queue stopped")
```

## Step 5: Update Endpoints

### Single TTS Endpoint

```python
@app.post("/tts")
async def single_tts(request: TTSRequest):
    """Single text-to-speech conversion with dynamic batching"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    try:
        # Step 1: Convert text to phonemes (individual G2P call)
        phonemes = await text_to_phonemes(request.text, request.language)
        
        # Step 2: Submit to batch queue or process directly
        if batch_queue and config.enable_kokoro_batching:
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
                "X-Batching-Enabled": str(config.enable_kokoro_batching)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
```

### Batch TTS Endpoint

```python
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
            task = text_to_phonemes(req.text, req.language)
            g2p_tasks.append(task)
        
        phonemes_list = await asyncio.gather(*g2p_tasks)
        
        # Step 2: Submit all to batch queue or process directly
        if batch_queue and config.enable_kokoro_batching:
            # Use dynamic batching
            kokoro_tasks = []
            for req, phonemes in zip(batch_request.requests, phonemes_list):
                task = batch_queue.submit_for_batching(
                    phonemes=phonemes,
                    voice=req.voice,
                    speed=req.speed
                )
                kokoro_tasks.append(task)
            
            audio_results = await asyncio.gather(*kokoro_tasks)
        else:
            # Direct processing
            audio_results = []
            for req, phonemes in zip(batch_request.requests, phonemes_list):
                audio = pipeline.from_phonemes(phonemes, req.voice, req.speed)
                audio_results.append(audio)
        
        # Step 3: Create ZIP response
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, (req, audio_tensor) in enumerate(zip(batch_request.requests, audio_results)):
                try:
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
                "X-Batching-Enabled": str(config.enable_kokoro_batching)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
```

### Streaming TTS Endpoint

```python
@app.post("/tts/stream")
async def streaming_tts(request: StreamingTTSRequest, client_request: Request):
    """Streaming text-to-speech conversion with chunk-level batching"""
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
                    pause_duration = float(text_chunk[9:-2])
                    silence_samples = int(pause_duration * 24000)
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
                
                # Generate audio for text chunk
                try:
                    # Convert text chunk to phonemes
                    phonemes = await text_to_phonemes(text_chunk, request.language)
                    
                    # Submit to batch queue or process directly
                    if batch_queue and config.enable_kokoro_batching:
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
                        await asyncio.sleep(0.01)
                    
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
            "X-Batching-Enabled": str(config.enable_kokoro_batching)
        }
    )
```

## Step 6: Add Monitoring Endpoints

```python
@app.get("/health/batching")
async def batching_health():
    """Detailed health check for batching system"""
    if not batch_queue:
        return {"batching_enabled": False, "status": "disabled"}
    
    stats = batch_queue.get_stats()
    
    return {
        "batching_enabled": config.enable_kokoro_batching,
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
```

## Step 7: Testing and Validation

### Basic Testing

1. **Start the service**:
   ```bash
   cd kokoro_test/kokoro_server
   python tts_service.py
   ```

2. **Test single request**:
   ```bash
   curl -X POST "http://localhost:8880/tts" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world", "voice": "af_heart"}' \
     --output test.wav
   ```

3. **Test batching performance**:
   ```bash
   curl -X POST "http://localhost:8880/test/batching"
   ```

4. **Check batching health**:
   ```bash
   curl "http://localhost:8880/health/batching"
   ```

### Performance Validation

- **Single Request Latency**: Should be < 200ms additional overhead
- **Batch Efficiency**: 4 concurrent requests should be faster than 4 sequential
- **Queue Health**: Queue size should remain low under normal load
- **Error Handling**: Failed requests should not block the entire batch

## Step 8: Deployment Considerations

### Production Configuration

```bash
# High-throughput configuration
KOKORO_MAX_BATCH_SIZE=8
KOKORO_MAX_WAIT_MS=100

# Low-latency configuration  
KOKORO_MAX_BATCH_SIZE=2
KOKORO_MAX_WAIT_MS=25

# Disable for debugging
ENABLE_KOKORO_BATCHING=false
```

### Monitoring

Monitor these metrics in production:
- Average batch size (target: 2-4)
- Queue depth (should stay < 10)
- Processing latency (target: < 200ms overhead)
- Error rates per batch

### Troubleshooting

Common issues:
- **High queue depth**: Increase `KOKORO_MAX_BATCH_SIZE` or reduce `KOKORO_MAX_WAIT_MS`
- **High latency**: Reduce `KOKORO_MAX_WAIT_MS` or `KOKORO_MAX_BATCH_SIZE`
- **Low batch efficiency**: Increase `KOKORO_MAX_WAIT_MS` if load is low

## Summary

This implementation provides:
- **Simple shared queue** for all endpoints
- **10-50ms batching windows** for low latency
- **Automatic fallback** when batching is disabled
- **Comprehensive monitoring** and health checks
- **Thread pool isolation** for CPU-bound Kokoro inference
- **Graceful error handling** with per-request isolation

The system maintains full backward compatibility while providing significant performance improvements through intelligent batching of the expensive Kokoro model inference operations.