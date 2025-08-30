#!/usr/bin/env python3
from kokoro.safe_pipeline import SafePipeline
import soundfile as sf
from tts_components import (
    Config,
    text_to_phonemes,
    audio_to_wav_bytes
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
    
    texts = [
        "[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.",
        "Hello, world!"
    ]

    voices = [
        "af_heart",
        "am_adam"
    ]

    speeds = [
        1.0,
        1.0
    ]
    
    phonemes = []
    phonemes.append(await text_to_phonemes(
        texts[0],
        "a",  # Default to English
        g2p_session,
        config.g2p_url,
        config.g2p_timeout
    ))
    phonemes.append(await text_to_phonemes(
        texts[1],
        "a",  # Default to English
        g2p_session,
        config.g2p_url,
        config.g2p_timeout
    ))

    logger.info(f"G2P conversion successful: {len(phonemes)} phonemes")
    
    # Generate audio using the pipeline
    logger.info("Generating audio...")
    
    # Generate audio from phonemes
    logger.debug(f"Generating audio from phonemes using voice=af_heart, speed=1.0")
    audio_tensor = pipeline.from_phonemes(
        phonemes=phonemes,
        voices=voices,
        speeds=speeds
    )
    
    logger.debug(f"Audio generation successful: {len(audio_tensor)} samples")
    # Save the audio to a file
    audio_bytes = audio_to_wav_bytes(audio_tensor)
    output_file = "demo_output.wav"
    sf.write(output_file, audio_tensor, 24000)
    logger.info(f"Audio saved to {output_file}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())