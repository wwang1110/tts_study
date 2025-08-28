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

class SingleProcessBatchQueue:
    """Single-process batching queue using asyncio primitives"""
    
    def __init__(self,
                 max_batch_size: int = 4,
                 normal_queue_max_wait_ms: int = 100,
                 normal_queue_min_wait_ms: int = 30,
                 high_priority_queue_max_wait_ms: int = 20,
                 high_priority_queue_min_wait_ms: int = 10,
                 max_queue_size: int = 1000):
        self.max_batch_size = max_batch_size
        
        # Normal Priority Queue
        self.normal_queue_max_wait_ms = normal_queue_max_wait_ms
        self.normal_queue_min_wait_ms = normal_queue_min_wait_ms
        
        # High Priority Queue
        self.high_priority_queue_max_wait_ms = high_priority_queue_max_wait_ms
        self.high_priority_queue_min_wait_ms = high_priority_queue_min_wait_ms
        
        self.normal_queue = asyncio.Queue(maxsize=max_queue_size)
        self.high_priority_queue = asyncio.Queue(maxsize=max_queue_size)
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
    
    async def submit_for_batching(self, phonemes: str, voice: str, speed: float = 1.0, high_priority: bool = False) -> torch.FloatTensor:
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
            # Put in the appropriate queue (non-blocking)
            if high_priority:
                self.high_priority_queue.put_nowait(entry)
            else:
                self.normal_queue.put_nowait(entry)
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
        """Collect entries for a batch with adaptive timeout, prioritizing the high-priority queue."""
        batch = []
        
        # 1. Prioritize the high-priority queue
        while not self.high_priority_queue.empty() and len(batch) < self.max_batch_size:
            batch.append(self.high_priority_queue.get_nowait())
        
        # If we have a high-priority item, use the shorter timeout
        is_high_priority_batch = len(batch) > 0
        
        # 2. If the batch is not full, check the normal queue
        if len(batch) < self.max_batch_size:
            try:
                # Wait for the first item from the normal queue if the batch is empty
                if not batch:
                    first_entry = await asyncio.wait_for(self.normal_queue.get(), timeout=0.1)
                    batch.append(first_entry)

                # Determine wait time based on whether we already have high-priority items
                if is_high_priority_batch:
                    wait_time_ms = self.high_priority_queue_max_wait_ms if len(batch) == 1 else self.high_priority_queue_min_wait_ms
                else:
                    wait_time_ms = self.normal_queue_max_wait_ms if len(batch) == 1 else self.normal_queue_min_wait_ms
                
                deadline = time.time() + (wait_time_ms / 1000.0)
                
                # Fill the rest of the batch from the normal queue
                while len(batch) < self.max_batch_size and time.time() < deadline:
                    remaining_time = deadline - time.time()
                    if remaining_time <= 0:
                        break
                    try:
                        entry = await asyncio.wait_for(self.normal_queue.get(), timeout=remaining_time)
                        batch.append(entry)
                    except asyncio.TimeoutError:
                        break
            except asyncio.TimeoutError:
                pass # It's okay if the normal queue is empty

        return batch
    
    async def _process_batch(self, batch: List[BatchEntry]):
        """Process batch and set results on Futures"""
        processing_start_time = time.time()
        
        # Log queue wait times
        for entry in batch:
            wait_duration_ms = (processing_start_time - entry.arrival_time) * 1000
            logger.info(f"Request {entry.id} waited in queue for {wait_duration_ms:.2f}ms")

        try:
            # Group by voice for efficient processing
            voice_groups = self._group_by_voice(batch)
            
            # Process each voice group
            for voice, entries in voice_groups.items():
                await self._process_voice_group(voice, entries)
            
            # Update metrics
            batch_time_ms = (time.time() - processing_start_time) * 1000
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
            "current_normal_queue_size": self.normal_queue.qsize(),
            "current_high_priority_queue_size": self.high_priority_queue.qsize(),
            "worker_running": self.running
        }