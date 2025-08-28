# Streaming TTS with Bypass Batching for First Chunks

## Simple Approach: Direct Pipeline for First Two Chunks

The cleanest implementation is to **completely bypass the batch queue** for the first two chunks in streaming mode, then use normal batching for subsequent chunks.

## Updated Streaming TTS Endpoint

```python
@app.post("/tts/stream")
async def streaming_tts_with_bypass(request: StreamingTTSRequest, client_request: Request):
    """Streaming TTS with direct processing for first 2 chunks, batching for rest"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    # Generate unique stream ID for tracking
    stream_id = str(uuid.uuid4())
    
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
                    
                    # BYPASS LOGIC: First 2 chunks go directly to pipeline
                    if chunk_count < 2:
                        # Direct processing - completely bypass batch queue
                        logger.info(f"Stream {stream_id} chunk {chunk_count}: DIRECT processing")
                        
                        audio_tensor = pipeline.from_phonemes(
                            phonemes=phonemes,
                            voice=request.voice,
                            speed=request.speed
                        )
                    else:
                        # Subsequent chunks use batch queue if available
                        if batch_queue and config.enable_kokoro_batching:
                            logger.info(f"Stream {stream_id} chunk {chunk_count}: BATCHED processing")
                            
                            audio_tensor = await batch_queue.submit_for_batching(
                                phonemes=phonemes,
                                voice=request.voice,
                                speed=request.speed
                            )
                        else:
                            # Fallback to direct processing if batching disabled
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
            "X-Stream-ID": stream_id,
            "X-Streaming-Mode": "bypass-first-chunks"
        }
    )
```

## Configuration for Bypass Behavior

Add to `.env.example`:

```bash
# Streaming Bypass Configuration
STREAMING_BYPASS_CHUNKS=2  # Number of chunks to process directly (bypass batching)
```

Update Config class:

```python
class Config:
    def __init__(self):
        # ... existing config ...
        
        # Streaming bypass configuration
        self.streaming_bypass_chunks = int(os.getenv("STREAMING_BYPASS_CHUNKS", "2"))
```

## More Flexible Implementation with Configuration

```python
@app.post("/tts/stream")
async def streaming_tts_configurable_bypass(request: StreamingTTSRequest, client_request: Request):
    """Streaming TTS with configurable bypass for first N chunks"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="TTS pipeline not ready")
    
    stream_id = str(uuid.uuid4())
    bypass_chunks = config.streaming_bypass_chunks  # Configurable number
    
    async def generate_audio_stream():
        try:
            chunk_count = 0
            
            async for text_chunk in simple_smart_split(request.text):
                if await client_request.is_disconnected():
                    break
                
                # Handle pause chunks (same as before)
                if text_chunk.startswith("__PAUSE__") and text_chunk.endswith("__"):
                    # ... pause handling code ...
                    continue
                
                try:
                    # Convert text chunk to phonemes
                    phonemes = await text_to_phonemes(text_chunk, request.language)
                    
                    # Bypass logic: First N chunks bypass batching completely
                    if chunk_count < bypass_chunks:
                        # DIRECT PROCESSING - No queue, no batching, immediate execution
                        start_time = time.time()
                        
                        audio_tensor = pipeline.from_phonemes(
                            phonemes=phonemes,
                            voice=request.voice,
                            speed=request.speed
                        )
                        
                        process_time = (time.time() - start_time) * 1000
                        logger.info(f"Stream {stream_id} chunk {chunk_count}: "
                                  f"BYPASSED batching, processed in {process_time:.1f}ms")
                    else:
                        # BATCHED PROCESSING - Use batch queue for efficiency
                        if batch_queue and config.enable_kokoro_batching:
                            start_time = time.time()
                            
                            audio_tensor = await batch_queue.submit_for_batching(
                                phonemes=phonemes,
                                voice=request.voice,
                                speed=request.speed
                            )
                            
                            process_time = (time.time() - start_time) * 1000
                            logger.info(f"Stream {stream_id} chunk {chunk_count}: "
                                      f"BATCHED processing, processed in {process_time:.1f}ms")
                        else:
                            # Fallback to direct processing
                            audio_tensor = pipeline.from_phonemes(
                                phonemes=phonemes,
                                voice=request.voice,
                                speed=request.speed
                            )
                    
                    # Convert and stream audio (same as before)
                    if request.format.lower() == "wav":
                        audio_bytes = audio_to_wav_bytes(audio_tensor)
                    else:
                        audio_bytes = audio_to_pcm_bytes(audio_tensor)
                    
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
    
    return StreamingResponse(
        generate_audio_stream(),
        media_type="audio/wav" if request.format.lower() == "wav" else "audio/pcm",
        headers={
            "Content-Disposition": f"attachment; filename=tts_stream.{request.format}",
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
            "X-Stream-ID": stream_id,
            "X-Streaming-Mode": f"bypass-first-{bypass_chunks}-chunks",
            "X-Bypass-Chunks": str(bypass_chunks)
        }
    )
```

## Simple Batch Queue (No Streaming Complexity)

Since streaming bypasses the queue for first chunks, we can use the simpler GPU-aware batch queue:

```python
# Use the GPUAwareBatchQueue from GPU_BATCHING_IMPLEMENTATION.md
# No need for dual queues or streaming-specific logic in the batch queue

from batching import GPUAwareBatchQueue  # Simple version

batch_queue: Optional[GPUAwareBatchQueue] = None

@app.on_event("startup")
async def startup_event():
    """Initialize with simple GPU-aware batching"""
    global pipeline, batch_queue, config
    
    try:
        config = Config()
        pipeline = SafePipeline()
        
        if config.enable_kokoro_batching:
            # Simple batch queue - streaming handles bypass logic
            batch_queue = GPUAwareBatchQueue(
                max_batch_size=config.kokoro_max_batch_size,
                max_wait_ms=config.kokoro_max_wait_ms,
                min_wait_ms=config.kokoro_min_wait_ms,
                max_queue_size=config.max_queue_size
            )
            batch_queue.set_pipeline(pipeline)
            await batch_queue.start_worker()
            
            print(f"✅ GPU-aware batching enabled")
            print(f"   Streaming: first {config.streaming_bypass_chunks} chunks bypass batching")
        
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        raise
```

## Performance Characteristics

### **Bypass Processing (Chunks 0-1)**
- **Latency**: ~50-100ms (direct pipeline call)
- **No Queue Overhead**: Zero batching delay
- **Immediate Execution**: Processed as soon as phonemes are ready
- **GPU Safe**: Direct call in main thread

### **Batched Processing (Chunks 2+)**
- **Latency**: ~60-150ms (includes 10-50ms batching window)
- **Efficiency**: Benefits from voice grouping and reduced GPU kernel launches
- **Throughput**: Better overall throughput for remaining chunks

### **Overall Streaming Experience**
- **Time-to-First-Audio**: Minimized by bypassing batching for first chunks
- **Sustained Performance**: Later chunks benefit from batching efficiency
- **Best of Both Worlds**: Fast start + efficient continuation

## Monitoring

Add endpoint to track bypass vs batched processing:

```python
@app.get("/health/streaming")
async def streaming_health():
    """Health check for streaming with bypass info"""
    return {
        "bypass_chunks_config": config.streaming_bypass_chunks,
        "batching_enabled": config.enable_kokoro_batching,
        "batch_queue_stats": batch_queue.get_stats() if batch_queue else None,
        "streaming_mode": "bypass-first-chunks"
    }
```

This approach is much cleaner:
1. **First N chunks**: Direct `pipeline.from_phonemes()` call (bypass everything)
2. **Remaining chunks**: Normal batch queue processing
3. **Simple Logic**: Easy to understand and maintain
4. **Configurable**: Adjust number of bypass chunks via environment variable
5. **No Queue Complexity**: Batch queue doesn't need streaming-specific logic

The bypass approach gives you the absolute minimum latency for the first chunks while still benefiting from batching efficiency for the bulk of the content.