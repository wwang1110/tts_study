#!/usr/bin/env python3
from kokoro.safe_pipeline import SafePipeline
import soundfile as sf
from tts_components import (
    Config,
    text_to_phonemes,
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main demo function"""
    # Global configuration and pipeline
    config = Config()
    
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=config.g2p_timeout)
    g2p_session = aiohttp.ClientSession(timeout=timeout)

    pipeline = SafePipeline(cache_dir="./.cache")
    
    text = '''
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''
    
    # Convert text to phonemes using G2P service
    logger.info(f"Converting text to phonemes: '{text[:50]}...'")
    
    # For demo purposes, use synchronous fallback (no aiohttp session)
    phonemes = await text_to_phonemes(
        text,
        "a",  # Default to English
        g2p_session,
        config.g2p_url,
        config.g2p_timeout
    )
    logger.info(f"G2P conversion successful: {len(phonemes)} phonemes")
    
    # Generate audio using the pipeline
    logger.info("Generating audio...")
    
    # Generate audio from phonemes
    logger.debug(f"Generating audio from phonemes using voice=af_heart, speed=1.0")
    audio_tensor = pipeline.from_phonemes(
        phonemes=phonemes,
        voice="af_heart",
        speed=1.0
    )
    
    logger.debug(f"Audio generation successful: {len(audio_tensor)} samples")
    # Save the audio to a file
    output_file = "demo_batch.wav"
    sf.write(output_file, audio_tensor, 24000)
    logger.info(f"Audio saved to {output_file}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())