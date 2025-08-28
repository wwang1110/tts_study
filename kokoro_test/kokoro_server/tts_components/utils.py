import io
import re
import wave
import numpy as np
import logging
from typing import Optional, AsyncGenerator

logger = logging.getLogger(__name__)

async def simple_smart_split(
    text: str,
    max_tokens: int,
    first_chunk_max_tokens: int,
) -> AsyncGenerator[str, None]:
    """
    Optimized version of smart_split that prioritizes fast time to first token.
    
    The first chunk is kept small (25-50 tokens, ideally one sentence) for faster
    initial audio generation, while subsequent chunks use normal sizing for quality.
    
    Args:
        text: Input text to split
        max_tokens: Maximum tokens per chunk for non-first chunks
        first_chunk_max_tokens: Maximum tokens for the first chunk
        
    Yields:
        Text chunks (first chunk optimized for speed, rest for quality)
    """
    # Simple approximation: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    first_chunk_max_chars = first_chunk_max_tokens * 4
    
    logger.debug(f"First chunk max chars: {first_chunk_max_chars}, subsequent max chars: {max_chars}")
    
    # Handle pause tags first
    pause_pattern = re.compile(r'\[pause:(\d+(?:\.\d+)?)s\]', re.IGNORECASE)
    parts = pause_pattern.split(text)
    
    is_first_text_chunk = True
    
    for i, part in enumerate(parts):
        if i % 2 == 1:  # This is a pause duration
            # Yield pause as special marker
            yield f"__PAUSE__{part}__"
            continue
            
        if not part.strip():
            continue
            
        # Split text part into sentences
        sentences = re.split(r'([.!?;:])\s*', part)
        
        # Handle first chunk specially for ULTRA-FAST time to first token
        if is_first_text_chunk and sentences:
            is_first_text_chunk = False
            
            # Try to get the first sentence
            first_sentence = sentences[0].strip() if sentences[0] else ""
            first_punct = sentences[1] if len(sentences) > 1 else ""
            
            if first_sentence:
                first_full_sentence = first_sentence + first_punct
                
                # ULTRA-AGGRESSIVE first chunk optimization - ALWAYS prioritize speed
                # Never process more than first_chunk_max_chars, regardless of sentence boundaries
                logger.debug(f"First sentence length: {len(first_full_sentence)}, max allowed: {first_chunk_max_chars}")
                logger.debug(f"First sentence: '{first_full_sentence}'")
                
                if len(first_full_sentence) <= first_chunk_max_chars:
                    # Perfect! First sentence fits in first chunk
                    logger.debug("First sentence fits within limits")
                    yield first_full_sentence.strip()
                    remaining_sentences = sentences[2:] if len(sentences) > 2 else []
                else:
                    logger.debug("First sentence too long, forcing truncation")
                    # FORCE ultra-small first chunk regardless of sentence structure
                    words = first_sentence.split()
                    truncated_sentence = ""
                    
                    # AGGRESSIVE: Take maximum 5-8 words for ultra-fast first response
                    max_first_words = min(8, len(words))  # Never more than 8 words
                    
                    for i, word in enumerate(words[:max_first_words]):
                        test_sentence = truncated_sentence + (" " if truncated_sentence else "") + word
                        if len(test_sentence) <= first_chunk_max_chars:
                            truncated_sentence = test_sentence
                        else:
                            break
                    
                    # Ensure we have at least 2 words, even if it slightly exceeds limit
                    if not truncated_sentence and len(words) >= 2:
                        truncated_sentence = f"{words[0]} {words[1]}"
                    elif not truncated_sentence:
                        truncated_sentence = words[0] if words else first_sentence[:first_chunk_max_chars]
                    
                    # Yield ultra-small first chunk for maximum speed (no punctuation to avoid incomplete sentences)
                    yield truncated_sentence.strip()
                    
                    # Put ALL remaining content back for normal processing
                    remaining_words = first_sentence[len(truncated_sentence):].strip()
                    if remaining_words:
                        # Reconstruct the remaining sentence with punctuation
                        remaining_sentences = [remaining_words + first_punct] + sentences[2:]
                    else:
                        remaining_sentences = sentences[2:] if len(sentences) > 2 else []
                
                # Process remaining sentences with normal chunking logic
                if remaining_sentences:
                    current_chunk = ""
                    for j in range(0, len(remaining_sentences), 2):
                        sentence = remaining_sentences[j].strip() if j < len(remaining_sentences) else ""
                        punct = remaining_sentences[j + 1] if j + 1 < len(remaining_sentences) else ""
                        
                        if not sentence:
                            continue
                            
                        full_sentence = sentence + punct
                        
                        # Check if adding this sentence exceeds max_chars
                        if len(current_chunk) + len(full_sentence) > max_chars:
                            # Yield current chunk and start new one
                            if current_chunk.strip():
                                yield current_chunk.strip()
                            current_chunk = full_sentence
                        else:
                            # Add to current chunk
                            if current_chunk:
                                current_chunk += " " + full_sentence
                            else:
                                current_chunk = full_sentence
                    
                    # Don't forget the last chunk
                    if current_chunk.strip():
                        yield current_chunk.strip()
            continue
        
        # Normal processing for non-first chunks
        current_chunk = ""
        for j in range(0, len(sentences), 2):
            sentence = sentences[j].strip()
            punct = sentences[j + 1] if j + 1 < len(sentences) else ""
            
            if not sentence:
                continue
                
            full_sentence = sentence + punct
            
            # Check if adding this sentence exceeds max_chars
            if len(current_chunk) + len(full_sentence) > max_chars:
                # Yield current chunk and start new one
                if current_chunk.strip():
                    yield current_chunk.strip()
                current_chunk = full_sentence
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + full_sentence
                else:
                    current_chunk = full_sentence
        
        # Don't forget the last chunk
        if current_chunk.strip():
            yield current_chunk.strip()

def audio_to_wav_bytes(audio_tensor, sample_rate=24000):
    """Convert audio tensor to WAV bytes"""
    logger.debug(f"Converting audio to WAV: sample_rate={sample_rate}")
    
    # Convert to numpy if it's a tensor
    if hasattr(audio_tensor, 'numpy'):
        audio_np = audio_tensor.numpy()
        logger.debug(f"Converted tensor to numpy: shape={audio_np.shape}, dtype={audio_np.dtype}")
    else:
        audio_np = audio_tensor
        logger.debug(f"Using numpy array: shape={audio_np.shape}, dtype={audio_np.dtype}")
    
    # Ensure int16 format
    original_dtype = audio_np.dtype
    if audio_np.dtype != np.int16:
        if audio_np.dtype == np.float32:
            audio_np = (audio_np * 32767).astype(np.int16)
            logger.debug(f"Converted float32 to int16: {original_dtype} -> {audio_np.dtype}")
        else:
            audio_np = audio_np.astype(np.int16)
            logger.debug(f"Converted {original_dtype} to int16")
    else:
        logger.debug("Audio already in int16 format")
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_np.tobytes())
    
    wav_bytes = wav_buffer.getvalue()
    duration_seconds = len(audio_np) / sample_rate
    logger.info(f"WAV created: {len(wav_bytes):,} bytes, duration={duration_seconds:.2f}s, samples={len(audio_np):,}")
    
    return wav_bytes

def create_wav_header(data_size, sample_rate=24000, channels=1, bits_per_sample=16):
    """Create WAV header for streaming"""
    # WAV header structure
    header = bytearray(44)
    
    # RIFF header
    header[0:4] = b'RIFF'
    header[4:8] = (36 + data_size).to_bytes(4, 'little')  # File size - 8
    header[8:12] = b'WAVE'
    
    # fmt subchunk
    header[12:16] = b'fmt '
    header[16:20] = (16).to_bytes(4, 'little')  # Subchunk1Size
    header[20:22] = (1).to_bytes(2, 'little')   # AudioFormat (PCM)
    header[22:24] = channels.to_bytes(2, 'little')  # NumChannels
    header[24:28] = sample_rate.to_bytes(4, 'little')  # SampleRate
    header[28:32] = (sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little')  # ByteRate
    header[32:34] = (channels * bits_per_sample // 8).to_bytes(2, 'little')  # BlockAlign
    header[34:36] = bits_per_sample.to_bytes(2, 'little')  # BitsPerSample
    
    # data subchunk
    header[36:40] = b'data'
    header[40:44] = data_size.to_bytes(4, 'little')  # Subchunk2Size
    
    return bytes(header)

def audio_to_pcm_bytes(audio_tensor):
    """Convert audio tensor to raw PCM bytes"""
    logger.debug("Converting audio to PCM")
    
    if hasattr(audio_tensor, 'numpy'):
        audio_np = audio_tensor.numpy()
        logger.debug(f"Converted tensor to numpy: shape={audio_np.shape}, dtype={audio_np.dtype}")
    else:
        audio_np = audio_tensor
        logger.debug(f"Using numpy array: shape={audio_np.shape}, dtype={audio_np.dtype}")
    
    original_dtype = audio_np.dtype
    if audio_np.dtype != np.int16:
        if audio_np.dtype == np.float32:
            audio_np = (audio_np * 32767).astype(np.int16)
            logger.debug(f"Converted float32 to int16: {original_dtype} -> {audio_np.dtype}")
        else:
            audio_np = audio_np.astype(np.int16)
            logger.debug(f"Converted {original_dtype} to int16")
    else:
        logger.debug("Audio already in int16 format")
    
    pcm_bytes = audio_np.tobytes()
    duration_seconds = len(audio_np) / 24000  # Assuming 24kHz sample rate
    logger.info(f"PCM created: {len(pcm_bytes):,} bytes, duration={duration_seconds:.2f}s, samples={len(audio_np):,}")
    
    return pcm_bytes