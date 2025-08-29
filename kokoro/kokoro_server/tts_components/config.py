import os

class Config:
    """Service configuration"""
    def __init__(self):
        import logging
        logger = logging.getLogger(__name__)

        self.g2p_url = os.getenv("G2P_SERVICE_URL", "http://localhost:5000")
        self.g2p_timeout = int(os.getenv("G2P_TIMEOUT", "5"))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "32"))
        self.max_tokens_per_chunk = int(os.getenv("MAX_TOKENS_PER_CHUNK", "100"))
        self.first_chunk_max_tokens = int(os.getenv("FIRST_CHUNK_MAX_TOKENS", "15"))
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8880"))

        # Dynamic Batching Configuration
        self.dynamic_batching = os.getenv("ENABLE_KOKORO_BATCHING", "true").lower() == "true"
        self.kokoro_max_batch_size = int(os.getenv("KOKORO_MAX_BATCH_SIZE", "4"))
        
        # Normal Priority Queue
        self.normal_queue_max_wait_ms = int(os.getenv("NORMAL_QUEUE_MAX_WAIT_MS", "100"))
        self.normal_queue_min_wait_ms = int(os.getenv("NORMAL_QUEUE_MIN_WAIT_MS", "30"))
        
        # High Priority Queue (for first streaming chunk)
        self.high_priority_queue_max_wait_ms = int(os.getenv("HIGH_PRIORITY_QUEUE_MAX_WAIT_MS", "20"))
        self.high_priority_queue_min_wait_ms = int(os.getenv("HIGH_PRIORITY_QUEUE_MIN_WAIT_MS", "10"))
        
        self.max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", "1000"))

