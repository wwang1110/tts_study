#!/usr/bin/env python3
"""
Two-Thread Dynamic Batching System for Kokoro TTS Server
Optimized architecture: HTTP Threads -> Single Queue+GPU Worker Thread
"""

import threading
import queue
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import torch

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Request entry for the batch queue"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phonemes: str = ""
    voice: str = ""
    speed: float = 1.0
    result_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[torch.Tensor] = None
    error: Optional[Exception] = None
    arrival_time: float = field(default_factory=time.time)
    queue_duration_ms: float = 0.0
    inference_duration_ms: float = 0.0
    
    def phoneme_length(self) -> int:
        return len(self.phonemes)
    
    def is_expired(self, timeout_seconds: float = 10.0) -> bool:
        return (time.time() - self.arrival_time) > timeout_seconds

class QueueGPUWorkerThread:
    """
    Single worker thread that handles both queue management and GPU processing
    Optimized for low latency and high throughput with dynamic batching
    """
    
    def __init__(self, 
                 max_batch_size: int = 4,
                 max_wait_ms: int = 50,
                 min_wait_ms: int = 10,
                 max_queue_size: int = 1000):
        
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds
        self.min_wait_ms = min_wait_ms / 1000.0
        
        # Thread-safe queue for incoming requests
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        
        # Device detection and setup
        self.device = self._detect_device()
        self.pipeline = None
        
        # Thread control
        self.running = False
        self.worker_thread = None
        
        # Metrics
        self.total_batches = 0
        self.total_requests = 0
        self.batch_sizes = []
        self._stats_lock = threading.Lock()
        
        logger.info(f"QueueGPUWorkerThread initialized with device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detect best available device with GPU priority"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def set_pipeline(self, pipeline):
        """Set the SafePipeline instance"""
        self.pipeline = pipeline
        logger.info(f"Pipeline set for QueueGPUWorkerThread")
    
    def start(self):
        """Start the worker thread"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("QueueGPUWorkerThread started")
    
    def stop(self):
        """Stop the worker thread"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            # Add sentinel to wake up the thread
            try:
                self.request_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            self.worker_thread.join(timeout=5.0)
        logger.info("QueueGPUWorkerThread stopped")
    
    def submit_for_batching(self, phonemes: str, voice: str, speed: float = 1.0) -> tuple[torch.Tensor, float, float]:
        """
        Submit a request for batching (called by HTTP threads)
        Returns the result after processing
        """
        if not self.running:
            raise RuntimeError("QueueGPUWorkerThread not running")
        
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        # Create request with threading.Event for synchronization
        request = BatchRequest(
            phonemes=phonemes,
            voice=voice,
            speed=speed
        )
        
        try:
            # Put request in queue (blocks if full)
            self.request_queue.put(request, timeout=5.0)
        except queue.Full:
            raise RuntimeError("Request queue is full - service overloaded")
        
        # Wait for result from worker thread
        if not request.result_event.wait(timeout=10.0):
            raise RuntimeError("Request processing timeout")
        
        # Check for errors
        if request.error:
            raise request.error
        
        if request.result is None:
            raise RuntimeError("Request processing failed - no result")
        
        return request.result, request.queue_duration_ms, request.inference_duration_ms
    
    def _worker_loop(self):
        """
        Main worker loop - handles both queue management and GPU processing
        """
        logger.info("QueueGPUWorkerThread worker loop started")
        
        # Initialize CUDA context in this thread if using GPU
        if self.device == 'cuda':
            try:
                torch.cuda.set_device(0)  # Set default GPU
                # Create a dummy tensor to initialize CUDA context
                _ = torch.zeros(1).to(self.device)
                logger.info("CUDA context initialized in QueueGPUWorkerThread")
            except Exception as e:
                logger.warning(f"Failed to initialize CUDA context, falling back to CPU: {e}")
                self.device = 'cpu'
        
        while self.running:
            try:
                # Collect batch with adaptive timeout
                batch = self._collect_batch()
                
                if batch:
                    # Process batch immediately (no handoff delay)
                    self._process_batch(batch)
                    
            except Exception as e:
                logger.error(f"QueueGPUWorkerThread worker error: {e}")
                time.sleep(0.1)  # Brief pause on error
        
        logger.info("QueueGPUWorkerThread worker loop ended")

    def _collect_batch(self) -> List[BatchRequest]:
        """Collect requests into a batch with adaptive timeout"""
        batch = []
        
        # Wait for first request (blocking with timeout)
        try:
            first_request = self.request_queue.get(timeout=0.1)
            if first_request is None:  # Sentinel for shutdown
                return []
            batch.append(first_request)
        except queue.Empty:
            return []  # No requests available
        
        # Calculate adaptive wait time based on current batch size
        wait_time = self.max_wait_ms if len(batch) == 1 else self.min_wait_ms
        deadline = time.time() + wait_time
        
        # Collect additional requests until batch is full or timeout
        while len(batch) < self.max_batch_size and time.time() < deadline:
            remaining_time = deadline - time.time()
            if remaining_time <= 0:
                break
            
            try:
                request = self.request_queue.get(timeout=remaining_time)
                if request is None:  # Sentinel for shutdown
                    break
                batch.append(request)
            except queue.Empty:
                break  # Timeout reached, process current batch
        
        return batch

    def _process_batch(self, batch: List[BatchRequest]):
        """Process batch directly with GPU/CPU inference"""
        processing_start_time = time.time()
        
        # Log queue wait times
        for request in batch:
            request.queue_duration_ms = (processing_start_time - request.arrival_time) * 1000
            logger.debug(f"Request {request.id} waited in queue for {request.queue_duration_ms:.2f}ms")

        try:
            # Group by voice for efficient processing
            voice_groups = self._group_by_voice(batch)
            
            # Process each voice group
            for voice, requests in voice_groups.items():
                self._process_voice_group(voice, requests)
            
            # Update metrics
            batch_time_ms = (time.time() - processing_start_time) * 1000
            self._update_metrics(len(batch), batch_time_ms)
            
            logger.debug(f"Processed batch: size={len(batch)}, time={batch_time_ms:.1f}ms, device={self.device}")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exception on all requests
            for request in batch:
                if not request.result_event.is_set():
                    request.error = e
                    request.result_event.set()

    def _group_by_voice(self, batch: List[BatchRequest]) -> Dict[str, List[BatchRequest]]:
        """Group batch requests by voice for efficient processing"""
        voice_groups = {}
        for request in batch:
            if request.voice not in voice_groups:
                voice_groups[request.voice] = []
            voice_groups[request.voice].append(request)
        return voice_groups

    def _process_voice_group(self, voice: str, requests: List[BatchRequest]):
        """Process requests with the same voice using GPU/CPU with true batching"""
        try:
            inference_start_time = time.time()
            
            # Prepare batch for inference
            phonemes_batch = []
            speeds_batch = []
            for req in requests:
                phonemes = req.phonemes
                if len(phonemes) > 510:
                    logger.warning(f"Phonemes too long ({len(phonemes)}), truncating")
                    phonemes = phonemes[:510]
                phonemes_batch.append(phonemes)
                speeds_batch.append(req.speed)

            # Perform true batch inference
            if self.pipeline is None:
                raise RuntimeError("Pipeline not initialized")
            
            # Note: This assumes `from_phonemes` can handle a batch of phonemes and speeds
            # This might require changes in the SafePipeline implementation
            audio_batch = self.pipeline.from_phonemes(
                phonemes=phonemes_batch,
                voice=voice,
                speed=speeds_batch
            )
            
            inference_duration_ms = (time.time() - inference_start_time) * 1000
            
            # Distribute results back to individual requests
            for i, request in enumerate(requests):
                try:
                    audio = audio_batch[i]
                    if hasattr(audio, 'cpu') and self.device != 'cpu':
                        audio = audio.cpu()
                    
                    request.result = audio
                    request.inference_duration_ms = inference_duration_ms / len(requests) # Average time
                    
                except Exception as e:
                    logger.error(f"Failed to assign result for request {request.id}: {e}")
                    request.result = torch.zeros(24000, dtype=torch.float32) # Fallback silence
                finally:
                    request.result_event.set()

        except Exception as e:
            logger.error(f"Batch inference failed for voice group {voice}: {e}")
            # Set exception on all requests in this group
            for request in requests:
                if not request.result_event.is_set():
                    request.error = e
                    request.result_event.set()
    
    def _update_metrics(self, batch_size: int, batch_time_ms: float):
        """Update batch processing metrics (thread-safe)"""
        with self._stats_lock:
            self.total_batches += 1
            self.total_requests += batch_size
            self.batch_sizes.append(batch_size)
            
            # Keep only last 100 batch sizes for rolling average
            if len(self.batch_sizes) > 100:
                self.batch_sizes.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics (thread-safe)"""
        with self._stats_lock:
            return {
                "total_batches": self.total_batches,
                "total_requests": self.total_requests,
                "avg_batch_size": sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0,
                "current_queue_size": self.request_queue.qsize(),
                "worker_running": self.running and self.worker_thread and self.worker_thread.is_alive(),
                "device": self.device,
                "pipeline_ready": self.pipeline is not None
            }

class ThreadBatchingHelper:
    """
    Main coordinator for the two-thread batching system
    HTTP Threads -> Single Queue+GPU Worker Thread
    """
    
    def __init__(self, 
                 max_batch_size: int = 4,
                 max_wait_ms: int = 50,
                 min_wait_ms: int = 10,
                 max_queue_size: int = 1000):
        
        # Initialize the single worker thread
        self.worker_thread = QueueGPUWorkerThread(
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            min_wait_ms=min_wait_ms,
            max_queue_size=max_queue_size
        )
        
        self.running = False
    
    def set_pipeline(self, pipeline):
        """Set the SafePipeline instance"""
        self.worker_thread.set_pipeline(pipeline)
    
    def start(self):
        """Start the batching system"""
        if not self.running:
            self.running = True
            self.worker_thread.start()
            logger.info("ThreadBatchingHelper started")
    
    def stop(self):
        """Stop the batching system"""
        if self.running:
            self.running = False
            self.worker_thread.stop()
            logger.info("ThreadBatchingHelper stopped")
    
    def submit_for_batching(self, phonemes: str, voice: str, speed: float = 1.0) -> tuple[torch.Tensor, float, float]:
        """
        Submit request for batching (called by HTTP threads)
        This is the main entry point from FastAPI handlers
        """
        if not self.running:
            raise RuntimeError("ThreadBatchingHelper not running")
        
        return self.worker_thread.submit_for_batching(phonemes, voice, speed)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        worker_stats = self.worker_thread.get_stats()
        
        return {
            "system_running": self.running,
            "worker_thread": worker_stats,
            "architecture": "two_thread_optimized"
        }
