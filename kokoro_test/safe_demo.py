#!/usr/bin/env python3
"""
License-Safe Kokoro Demo

This demo shows how to use Kokoro TTS without GPL dependencies.
Safe for production use in commercial applications.

Usage:
    # Method 1: Direct phonemes (recommended)
    python safe_demo.py --method phonemes
    
    # Method 2: G2P service (requires g2p_service.py running)
    python safe_demo.py --method service
"""

from safe_pipeline import SafePipeline, phonemes_to_audio
import soundfile as sf
import torch
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_direct_phonemes():
    """
    Demo using direct phonemes (no GPL dependencies).
    This is the recommended approach for production.
    """
    print("=== License-Safe Demo: Direct Phonemes ===")
    print("This method uses pre-converted phonemes and has no GPL dependencies.")
    
    # Initialize safe pipeline
    pipeline = SafePipeline()
    
    # Example phonemes for common phrases
    # These can be pre-computed offline using the G2P service
    phoneme_examples = {
        "hello_world": "həˈloʊ wɜrld",
        "kokoro_intro": "kˈOkəɹO ɪz ɐn ˈOpᵊnwˌAt tˌitˌiˈɛs mˈɑdᵊl",
        "simple_test": "ðɪs ɪz ə tˈɛst",
        "numbers": "wʌn tˈu θɹˈi fˈɔɹ fˈaɪv",
        "license_safe": "lˈaɪsəns sˈeɪf tˈɛkst tə spˈiʧ"
    }
    
    print(f"\nGenerating audio for {len(phoneme_examples)} examples...")
    
    for i, (name, phonemes) in enumerate(phoneme_examples.items()):
        print(f"\n{i+1}. {name}")
        print(f"   Phonemes: {phonemes}")
        
        try:
            # Generate audio
            audio = pipeline.from_phonemes(phonemes, voice='af_heart')
            
            # Save audio file
            filename = f"safe_demo_{name}.wav"
            sf.write(filename, audio.numpy(), 24000)
            print(f"   Saved: {filename}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n✅ Direct phonemes demo completed successfully!")
    print("This approach is completely license-safe for production use.")

def demo_g2p_service():
    """
    Demo using G2P service (GPL isolated in separate process).
    Requires g2p_service.py to be running.
    """
    print("=== License-Safe Demo: G2P Service ===")
    print("This method uses an external G2P service to isolate GPL dependencies.")
    
    # Check if G2P service is available
    try:
        from g2p_client import check_service_health
        service_url = "http://localhost:5001"
        if not check_service_health(service_url):
            print("❌ G2P service is not available!")
            print("Start it with: python g2p_service.py")
            return False
    except ImportError:
        print("❌ g2p_client not found. Make sure it's in the same directory.")
        return False
    
    # Initialize safe pipeline
    pipeline = SafePipeline()
    
    # Example texts with different languages
    text_examples = [
        ("Hello world, this is a test.", "en-US"),
        ("Kokoro is an open-weight TTS model.", "en-GB"),
        ("Hola mundo, esto es una prueba.", "es"),
        ("Bonjour le monde, ceci est un test.", "fr-FR"),
        ("Olá mundo, este é um teste.", "pt-BR")
    ]
    
    print(f"\nGenerating audio for {len(text_examples)} examples in multiple languages...")
    
    for i, (text, lang) in enumerate(text_examples):
        print(f"\n{i+1}. Text: {text}")
        print(f"   Language: {lang}")
        
        try:
            # Convert text to audio via G2P service with specified language
            audio = pipeline.from_text(text, voice='af_heart', lang=lang, g2p_url=service_url)
            
            # Save audio file
            filename = f"safe_demo_service_{lang}_{i+1}.wav"
            sf.write(filename, audio.numpy(), 24000)
            print(f"   Saved: {filename}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n✅ G2P service demo completed successfully!")
    print("The GPL code runs in a separate process, keeping production code license-safe.")
    return True

def demo_convenience_function():
    """Demo using the convenience function."""
    print("\n=== Convenience Function Demo ===")
    
    # Simple one-liner usage
    phonemes = "kənˈvinjəns fʌŋkʃən dˈɛmoʊ"
    audio = phonemes_to_audio(phonemes, voice='af_heart')
    
    filename = "safe_demo_convenience.wav"
    sf.write(filename, audio.numpy(), 24000)
    print(f"Generated: {filename}")

def main():
    parser = argparse.ArgumentParser(description="License-Safe Kokoro Demo")
    args = parser.parse_args()
    
    print("🎵 License-Safe Kokoro TTS Demo")
    print("=" * 50)
    print("This demo uses G2P service for license-safe text-to-speech synthesis.")
    print()
    
    try:
        # Always use G2P service method
        success = demo_g2p_service()
        if not success:
            print("\n❌ G2P service is not available!")
            print("Start it with: python kokoro_test/g2p_service.py")
            return 1
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed!")
        print("\nLicense compliance:")
        print("• G2P service: GPL isolated in separate process")
        print("• Production code: 100% GPL-free")
        print("• Safe for commercial use")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())