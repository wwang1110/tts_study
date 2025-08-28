import os

class Config:
    """Service configuration"""
    def __init__(self):
        import logging
        logger = logging.getLogger(__name__)

        self.g2p_url = os.getenv("G2P_SERVICE_URL", "http://localhost:8889")
        self.g2p_timeout = int(os.getenv("G2P_TIMEOUT", "5"))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "32"))
        self.max_tokens_per_chunk = int(os.getenv("MAX_TOKENS_PER_CHUNK", "100"))
        self.first_chunk_max_tokens = int(os.getenv("FIRST_CHUNK_MAX_TOKENS", "15"))
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8880"))

        logger.info("=" * 60)
        logger.info("🔧 Configuration Loaded:")
        logger.info(f"  G2P_SERVICE_URL: {self.g2p_url}")
        logger.info(f"  G2P_TIMEOUT: {self.g2p_timeout}")
        logger.info(f"  MAX_BATCH_SIZE: {self.max_batch_size}")
        logger.info(f"  MAX_TOKENS_PER_CHUNK: {self.max_tokens_per_chunk}")
        logger.info(f"  FIRST_CHUNK_MAX_TOKENS: {self.first_chunk_max_tokens}")
        logger.info(f"  HOST: {self.host}")
        logger.info(f"  PORT: {self.port}")
        logger.info("=" * 60)
