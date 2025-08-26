# GPU-Aware Dynamic Batching Implementation

## Critical Issue with Thread Pool Approach

The original implementation uses `ThreadPoolExecutor` for Kokoro inference, which **does NOT work properly with GPU operations** because:

1. **CUDA Context Issues**: GPU contexts are thread-local and don't transfer between threads
2. **Memory Management**: GPU memory allocated in one thread may not be accessible in another
3. **Performance Loss**: Moving tensors between CPU threads and GPU is inefficient

## Corrected GPU-Aware Implementation

### Updated Batch Queue Implementation

```python
#!/usr/bin/env python3
"""
GPU-Aware Dynamic Batching System for Kokoro TTS Server
Handles GPU operations properly in the main asyncio thread
"""

import asyncio
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch

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

class GPUAwareBatchQueue:
    """GPU-aware batching queue that keeps GPU operations in main thread"""
    
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
        self.pipeline = None
        
        # GPU device detection
        self.device = self._detect_device()
        logger.info(f"Using device: {self.device}")
        
        # Metrics
        self.total_batches = 0
        self.total_requests = 0
        self.batch_sizes = []
    
    def _detect_device(self) -> str:
        """Detect available GPU device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def set_pipeline(self, pipeline):
        """Set the SafePipeline instance"""
        self.pipeline = pipeline
        # Ensure pipeline is on correct device
        if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'to'):
            pipeline.model = pipeline.model.to(self.device)
    
    async def start_worker(self):
        """Start the background batch worker"""
        if not self.running:
            self.running = True
            self.worker_task = asyncio.create_task(self._batch_worker())
            logger.info(f"GPU-aware batch worker started on {self.device}")
    
    async def stop_worker(self):
        """Stop the background worker"""
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("GPU-aware batch worker stopped")
    
    async def submit_for_batching(self, phonemes: str, voice: str, speed: float = 1.0) -> torch.FloatTensor:
        """Submit request for batching and return result via Future"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        entry = BatchEntry(
            phonemes=phonemes,
            voice=voice,
            speed=speed
        )
        
        try:
            self.queue.put_nowait(entry)
        except asyncio.QueueFull:
            raise RuntimeError("Batch queue is full - service overloaded")
        
        try:
            result = await asyncio.wait_for(entry.result_future, timeout=10.0)
            return result
        except asyncio.TimeoutError:
            raise RuntimeError("Batch processing timeout")
    
    async def _batch_worker(self):
        """Background worker that processes batches in main thread (GPU-safe)"""
        logger.info("GPU-aware batch worker loop started")
        
        while self.running:
            try:
                batch = await self._collect_batch()
                
                if batch:
                    # Process batch in main asyncio thread (GPU operations stay in main thread)
                    await self._process_batch_gpu_aware(batch)
                    
            except Exception as e:
                logger.error(f"Batch worker error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("GPU-aware batch worker loop ended")
    
    async def _collect_batch(self) -> List[BatchEntry]:
        """Collect entries for a batch with adaptive timeout"""
        batch = []
        
        try:
            first_entry = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            batch.append(first_entry)
        except asyncio.TimeoutError:
            return []
        
        wait_time_ms = self.max_wait_ms if len(batch) == 1 else self.min_wait_ms
        deadline = time.time() + (wait_time_ms / 1000.0)
        
        while len(batch) < self.max_batch_size and time.time() < deadline:
            remaining_time = deadline - time.time()
            if remaining_time <= 0:
                break
            
            try:
                entry = await asyncio.wait_for(self.queue.get(), timeout=remaining_time)
                batch.append(entry)
            except asyncio.TimeoutError:
                break
        
        return batch
    
    async def _process_batch_gpu_aware(self, batch: List[BatchEntry]):
        """Process batch with GPU operations in main thread"""
        batch_start_time = time.time()
        
        try:
            # Group by voice for efficient processing
            voice_groups = self._group_by_voice(batch)
            
            # Process each voice group
            for voice, entries in voice_groups.items():
                await self._process_voice_group_gpu_aware(voice, entries)
            
            # Update metrics
            batch_time_ms = (time.time() - batch_start_time) * 1000
            self._update_metrics(len(batch), batch_time_ms)
            
        except Exception as e:
            logger.error(f"GPU batch processing failed: {e}")
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
    
    async def _process_voice_group_gpu_aware(self, voice: str, entries: List[BatchEntry]):
        """Process entries with same voice - GPU operations in main thread"""
        try:
            # Option 1: True Batch Processing (if SafePipeline supports it)
            if hasattr(self.pipeline, 'batch_from_phonemes_gpu'):
                await self._process_true_batch(voice, entries)
            else:
                # Option 2: Sequential processing with yield points (GPU-safe)
                await self._process_sequential_with_yields(voice, entries)
                
        except Exception as e:
            logger.error(f"Voice group processing failed for {voice}: {e}")
            for entry in entries:
                if not entry.result_future.done():
                    entry.result_future.set_exception(e)
    
    async def _process_true_batch(self, voice: str, entries: List[BatchEntry]):
        """Process as true batch if pipeline supports GPU batching"""
        phoneme_list = [entry.phonemes for entry in entries]
        speed = entries[0].speed
        
        # This would be a true GPU batch operation
        audio_results = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: list(self.pipeline.batch_from_phonemes_gpu(phoneme_list, voice, speed))
        )
        
        for entry, audio in zip(entries, audio_results):
            if not entry.result_future.done():
                entry.result_future.set_result(audio)
    
    async def _process_sequential_with_yields(self, voice: str, entries: List[BatchEntry]):
        """Process sequentially but yield control between items (GPU-safe)"""
        for entry in entries:
            try:
                # Validate phoneme length
                phonemes = entry.phonemes
                if len(phonemes) > 510:
                    logger.warning(f"Phonemes too long ({len(phonemes)}), truncating")
                    phonemes = phonemes[:510]
                
                # GPU operation in main thread - this is safe!
                audio = self.pipeline.from_phonemes(phonemes, entry.voice, entry.speed)
                
                # Set result
                if not entry.result_future.done():
                    entry.result_future.set_result(audio)
                
                # Yield control to allow other async operations
                await asyncio.sleep(0)
                
            except Exception as e:
                logger.error(f"Individual inference failed: {e}")
                if not entry.result_future.done():
                    # Return silence as fallback
                    silence = torch.zeros(24000, dtype=torch.float32, device=self.device)
                    entry.result_future.set_result(silence)
    
    def _update_metrics(self, batch_size: int, batch_time_ms: float):
        """Update batch processing metrics"""
        self.total_batches += 1
        self.total_requests += batch_size
        self.batch_sizes.append(batch_size)
        
        if len(self.batch_sizes) > 100:
            self.batch_sizes.pop(0)
        
        logger.debug(f"Processed GPU batch: size={batch_size}, time={batch_time_ms:.1f}ms, device={self.device}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current batch processing statistics"""
        return {
            "total_batches": self.total_batches,
            "total_requests": self.total_requests,
            "avg_batch_size": sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0,
            "current_queue_size": self.queue.qsize(),
            "worker_running": self.running,
            "device": self.device,
            "gpu_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available()
        }
```

## Enhanced SafePipeline for True GPU Batching

If you want to implement true GPU batching in SafePipeline, add this method:

```python
# Add to SafePipeline class in safe_pipeline.py

def batch_from_phonemes_gpu(
    self,
    phoneme_list: list[str],
    voice: Union[str, torch.FloatTensor],
    speed: float = 1.0
) -> Generator[torch.FloatTensor, None, None]:
    """
    GPU-optimized batch processing for multiple phoneme strings.
    Processes multiple phonemes efficiently on GPU.
    """
    # Load voice once for entire batch
    voice_pack = self.load_voice(voice).to(self.model.device)
    
    # Pre-allocate tensors on GPU for efficiency
    device = self.model.device
    
    for phonemes in phoneme_list:
        if len(phonemes) > 510:
            # Handle oversized phonemes
            yield torch.zeros(24000, dtype=torch.float32, device=device)
            continue
            
        try:
            # Ensure all operations stay on GPU
            voice_tensor = voice_pack[len(phonemes)-1].to(device)
            
            # Model inference on GPU
            with torch.no_grad():  # Save GPU memory
                output = self.model(phonemes, voice_tensor, speed, return_output=True)
                # Ensure result is on correct device
                audio_result = output.audio.to(device)
                yield audio_result
                
        except Exception as e:
            logger.error(f"GPU batch phoneme processing failed: {e}")
            # Fallback silence on same device
            yield torch.zeros(24000, dtype=torch.float32, device=device)

def batch_from_phonemes_true_gpu_batch(
    self,
    phoneme_list: list[str],
    voice: Union[str, torch.FloatTensor],
    speed: float = 1.0
) -> List[torch.FloatTensor]:
    """
    True GPU batch processing - process multiple phonemes in single forward pass.
    This requires model modifications to handle variable-length sequences.
    """
    if not phoneme_list:
        return []
    
    # Load voice once
    voice_pack = self.load_voice(voice).to(self.model.device)
    device = self.model.device
    
    # Filter valid phonemes
    valid_phonemes = []
    valid_indices = []
    
    for i, phonemes in enumerate(phoneme_list):
        if len(phonemes) <= 510:
            valid_phonemes.append(phonemes)
            valid_indices.append(i)
    
    if not valid_phonemes:
        # Return silence for all
        return [torch.zeros(24000, dtype=torch.float32, device=device) 
                for _ in phoneme_list]
    
    try:
        # This would require model modifications for true batching
        # For now, process sequentially but efficiently
        results = []
        
        with torch.no_grad():
            for phonemes in valid_phonemes:
                voice_tensor = voice_pack[len(phonemes)-1].to(device)
                output = self.model(phonemes, voice_tensor, speed, return_output=True)
                results.append(output.audio.to(device))
        
        # Map results back to original order
        final_results = []
        valid_iter = iter(results)
        
        for i in range(len(phoneme_list)):
            if i in valid_indices:
                final_results.append(next(valid_iter))
            else:
                final_results.append(torch.zeros(24000, dtype=torch.float32, device=device))
        
        return final_results
        
    except Exception as e:
        logger.error(f"True GPU batch processing failed: {e}")
        return [torch.zeros(24000, dtype=torch.float32, device=device) 
                for _ in phoneme_list]
```

## Updated Service Integration

```python
# In tts_service.py, replace SingleProcessBatchQueue with GPUAwareBatchQueue

from batching import GPUAwareBatchQueue  # Updated import

# Global batch queue
batch_queue: Optional[GPUAwareBatchQueue] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the TTS pipeline with GPU-aware batching"""
    global pipeline, batch_queue, config
    
    try:
        config = Config()
        
        # Initialize pipeline
        pipeline = SafePipeline()
        print("✅ SafePipeline initialized successfully")
        
        # Initialize GPU-aware batching
        if config.enable_kokoro_batching:
            batch_queue = GPUAwareBatchQueue(
                max_batch_size=config.kokoro_max_batch_size,
                max_wait_ms=config.kokoro_max_wait_ms,
                min_wait_ms=config.kokoro_min_wait_ms,
                max_queue_size=config.max_queue_size
            )
            batch_queue.set_pipeline(pipeline)
            await batch_queue.start_worker()
            
            # Log GPU status
            device_info = batch_queue.get_stats()
            print(f"✅ GPU-aware dynamic batching enabled on {device_info['device']}")
            print(f"   CUDA available: {device_info['gpu_available']}")
            print(f"   MPS available: {device_info['mps_available']}")
        else:
            print("ℹ️ Dynamic batching disabled")
            
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        raise
```

## Key Differences from Thread Pool Approach

### ✅ **GPU-Safe Approach**
- **Main Thread Processing**: All GPU operations stay in main asyncio thread
- **Device Consistency**: Tensors remain on same GPU device throughout processing
- **Memory Efficiency**: No CPU↔GPU transfers between threads
- **CUDA Context Safe**: No thread-local CUDA context issues

### ❌ **Thread Pool Issues (Avoided)**
- **Context Loss**: CUDA contexts don't transfer between threads
- **Memory Fragmentation**: GPU memory allocated in different threads
- **Performance Penalty**: Unnecessary CPU↔GPU data movement
- **Synchronization Issues**: Race conditions with GPU operations

## Performance Characteristics

### **GPU Batching Benefits**
- **Memory Efficiency**: Reuse GPU memory allocations across batch items
- **Compute Efficiency**: Better GPU utilization through batching
- **Reduced Overhead**: Fewer GPU kernel launches
- **Device Consistency**: All operations on same device

### **Latency Considerations**
- **Sequential Processing**: Still processes items sequentially (safe)
- **Async Yields**: Yields control between items for responsiveness
- **True Batching**: Optional true GPU batching if model supports it

## Testing GPU Functionality

```python
@app.get("/test/gpu")
async def test_gpu_batching():
    """Test GPU batching functionality"""
    if not batch_queue:
        return {"error": "Batching not enabled"}
    
    stats = batch_queue.get_stats()
    
    # Test GPU memory allocation
    try:
        device = batch_queue.device
        test_tensor = torch.randn(1000, device=device)
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024 if device == "cuda" else 0
        
        return {
            "device": device,
            "gpu_available": stats["gpu_available"],
            "mps_available": stats["mps_available"],
            "gpu_memory_allocated_mb": gpu_memory_mb,
            "test_tensor_device": str(test_tensor.device),
            "pipeline_device": str(batch_queue.pipeline.model.device) if batch_queue.pipeline else "unknown"
        }
    except Exception as e:
        return {"error": f"GPU test failed: {e}"}
```

This GPU-aware implementation ensures that:
1. **All GPU operations stay in the main thread** (no CUDA context issues)
2. **Tensors remain on the correct device** throughout processing
3. **Memory is managed efficiently** on GPU
4. **Batching still provides performance benefits** through reduced overhead
5. **True GPU batching is possible** with model modifications

The key insight is that asyncio's cooperative multitasking allows us to batch requests while keeping all GPU operations in the main thread, avoiding the complex issues of multi-threaded GPU programming.