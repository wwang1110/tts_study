#!/usr/bin/env python3
"""
Streaming Test Client with Immediate Audio Playback
Based on test case 5 from streaming_test_client.py
Plays audio immediately when first chunk is received
"""

import asyncio
import aiohttp
import time
import json
import sys
import threading
import tempfile
import os
import subprocess
import platform
import struct
import wave
from typing import Dict, List, Optional

# Audio playback imports
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import simpleaudio as sa
    SIMPLEAUDIO_AVAILABLE = True
except ImportError:
    SIMPLEAUDIO_AVAILABLE = False

# System audio fallback
SYSTEM_AUDIO_AVAILABLE = True

class TTSStreamingClientWithPlayback:
    """Test client for streaming TTS service with immediate audio playback"""
    
    def __init__(self, base_url: str = "http://localhost:8880"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.audio_initialized = False
        self.audio_queue = []
        self.playback_thread = None
        self.playing = False
        self.total_chunks_expected = 0
        self.chunks_played_count = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.audio_initialized and PYGAME_AVAILABLE:
            self.playing = False
            if self.playback_thread:
                self.playback_thread.join(timeout=1.0)
            pygame.mixer.quit()
    
    def init_audio(self):
        """Initialize audio system - prioritize pygame"""
        if not PYGAME_AVAILABLE:
            print("❌ pygame not available. Please install with: pip install pygame")
            raise RuntimeError("pygame is required for reliable audio playback. Install with: pip install pygame")
        
        if not self.audio_initialized:
            try:
                # Use smaller buffer for lower latency
                pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=256)
                pygame.mixer.init()
                # Set up multiple channels for seamless playback
                pygame.mixer.set_num_channels(8)
                self.audio_initialized = True
                print("🔊 Audio system initialized with pygame")
                print(f"   Mixer settings: {pygame.mixer.get_init()}")
                print(f"   Available channels: {pygame.mixer.get_num_channels()}")
                return
            except Exception as e:
                print(f"❌ pygame initialization failed: {e}")
                raise RuntimeError(f"Failed to initialize pygame audio: {e}")
    
    def play_audio_with_system(self, temp_path: str, chunk_number: int):
        """Play audio using system commands as fallback"""
        try:
            system = platform.system().lower()
            if system == "darwin":  # macOS
                subprocess.run(["afplay", temp_path], check=True, capture_output=True)
            elif system == "linux":
                # Try multiple Linux audio players
                for player in ["aplay", "paplay", "play"]:
                    try:
                        subprocess.run([player, temp_path], check=True, capture_output=True)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                else:
                    raise RuntimeError("No suitable audio player found on Linux")
            elif system == "windows":
                # Use Windows Media Player or PowerShell
                try:
                    subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{temp_path}').PlaySync()"], 
                                 check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    # Fallback to start command
                    subprocess.run(["start", "/wait", temp_path], shell=True, check=True)
            else:
                raise RuntimeError(f"Unsupported system: {system}")
            
            print(f"🎵 Playing chunk {chunk_number} with system audio")
            
        except Exception as e:
            print(f"❌ System audio playback error: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def continuous_playback_worker(self):
        """Worker thread for seamless audio playback using Sound objects"""
        current_channel = 0
        active_sounds = []
        
        while self.playing or self.audio_queue:
            if self.audio_queue:
                try:
                    audio_data, chunk_number = self.audio_queue.pop(0)
                    
                    # Create temporary WAV file
                    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
                    os.close(temp_fd)
                    
                    # Create WAV from raw audio data
                    sample_rate = 22050
                    channels = 1
                    sample_width = 2
                    
                    with wave.open(temp_path, 'wb') as wav_file:
                        wav_file.setnchannels(channels)
                        wav_file.setsampwidth(sample_width)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_data)
                    
                    # Load as Sound object for better control
                    sound = pygame.mixer.Sound(temp_path)
                    
                    # Wait for previous sound to almost finish to minimize gaps
                    if active_sounds:
                        last_sound, last_channel = active_sounds[-1]
                        # Wait until the last sound is almost done (leave small overlap)
                        while last_channel.get_busy():
                            time.sleep(0.001)  # Very short sleep for precise timing
                    
                    # Play the sound on next available channel
                    channel = sound.play()
                    if channel:
                        active_sounds.append((sound, channel))
                        print(f"🎵 Playing chunk {chunk_number} seamlessly")
                        
                        # Clean up old finished sounds
                        active_sounds = [(s, c) for s, c in active_sounds if c.get_busy()]
                    
                    # Update played count
                    self.chunks_played_count += 1
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
                except Exception as e:
                    print(f"❌ Seamless playback error: {e}")
            else:
                time.sleep(0.001)  # Very short delay when queue is empty
        
        # Wait for all remaining sounds to finish
        while active_sounds:
            active_sounds = [(s, c) for s, c in active_sounds if c.get_busy()]
            if active_sounds:
                time.sleep(0.01)
        
        print(f"🎵 Seamless playback finished. Played {self.chunks_played_count} chunks.")
    
    def start_continuous_playback(self):
        """Start the continuous playback system"""
        if not self.playing:
            self.playing = True
            self.playback_thread = threading.Thread(target=self.continuous_playback_worker, daemon=True)
            self.playback_thread.start()
            print("🎵 Started continuous playback system")
    
    def add_audio_chunk(self, audio_data: bytes, chunk_number: int):
        """Add audio chunk to the playback queue"""
        self.audio_queue.append((audio_data, chunk_number))
        print(f"   Added chunk {chunk_number} to playback queue ({len(audio_data)} bytes, queue size: {len(self.audio_queue)})")
    
    def wait_for_playback_completion(self):
        """Wait for all queued audio to finish playing"""
        print(f"⏳ Waiting for {len(self.audio_queue)} remaining chunks to play...")
        
        # Stop adding new chunks but let existing ones finish
        self.playing = False
        
        # Wait for playback thread to finish processing all chunks
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()
        
        print(f"✅ All audio chunks have been played!")
    
    async def check_health(self) -> Dict:
        """Check if the TTS service is healthy"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise RuntimeError(f"Health check failed: {response.status}")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to TTS service: {e}")
    
    async def stream_tts_with_immediate_playback(self, text: str, voice: str = "af_heart", 
                                               language: str = "en-US", format: str = "wav",
                                               save_file: Optional[str] = None,
                                               min_chunk_size: int = 8192) -> Dict:
        """
        Stream TTS with immediate audio playback as chunks arrive
        Plays audio immediately when sufficient data is received
        """
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        # Initialize audio system
        self.init_audio()
        
        payload = {
            "text": text,
            "voice": voice,
            "language": language,
            "format": format
        }
        
        metrics = {
            "total_time": 0,
            "time_to_first_byte": 0,
            "time_to_first_playback": 0,
            "chunks_received": 0,
            "chunks_played": 0,
            "total_bytes": 0,
            "chunk_times": [],
            "text_length": len(text),
            "success": False,
            "saved_file": None,
            "playback_started": False
        }
        
        start_time = time.time()
        first_byte_time = None
        first_playback_time = None
        audio_buffer = bytearray()
        audio_data = bytearray()
        
        try:
            async with self.session.post(f"{self.base_url}/tts/stream", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Streaming failed: {response.status} - {error_text}")
                
                chunk_start_time = time.time()
                
                async for chunk in response.content.iter_chunked(1024):
                    if first_byte_time is None:
                        first_byte_time = time.time()
                        metrics["time_to_first_byte"] = first_byte_time - start_time
                        print(f"📡 First chunk received after {metrics['time_to_first_byte']:.3f}s")
                    
                    metrics["chunks_received"] += 1
                    metrics["total_bytes"] += len(chunk)
                    audio_buffer.extend(chunk)
                    audio_data.extend(chunk)
                    
                    # Play audio immediately when we have enough data
                    if not metrics["playback_started"] and len(audio_buffer) >= min_chunk_size:
                        first_playback_time = time.time()
                        metrics["time_to_first_playback"] = first_playback_time - start_time
                        metrics["playback_started"] = True
                        
                        print(f"🎵 Starting immediate playback after {metrics['time_to_first_playback']:.3f}s")
                        print(f"   Buffer size: {len(audio_buffer):,} bytes")
                        
                        # Start continuous playback system
                        self.start_continuous_playback()
                        
                        # Add the first chunk to playback queue
                        self.add_audio_chunk(bytes(audio_buffer), metrics["chunks_played"] + 1)
                        metrics["chunks_played"] += 1
                        
                        # Clear buffer for next chunk
                        audio_buffer.clear()
                    
                    # Continue adding subsequent chunks to the queue
                    elif metrics["playback_started"] and len(audio_buffer) >= min_chunk_size:
                        self.add_audio_chunk(bytes(audio_buffer), metrics["chunks_played"] + 1)
                        metrics["chunks_played"] += 1
                        audio_buffer.clear()
                    
                    chunk_time = time.time() - chunk_start_time
                    metrics["chunk_times"].append(chunk_time)
                    chunk_start_time = time.time()
                
                # Add any remaining audio in buffer to queue
                if len(audio_buffer) > 0:
                    if not metrics["playback_started"]:
                        self.start_continuous_playback()
                    self.add_audio_chunk(bytes(audio_buffer), metrics["chunks_played"] + 1)
                    metrics["chunks_played"] += 1
                
                metrics["total_time"] = time.time() - start_time
                metrics["success"] = True
                
                # Save complete audio to file if requested
                if save_file and audio_data:
                    with open(save_file, 'wb') as f:
                        f.write(audio_data)
                    metrics["saved_file"] = save_file
                    print(f"💾 Complete audio saved to: {save_file} ({len(audio_data):,} bytes)")
                
        except Exception as e:
            metrics["error"] = str(e)
        finally:
            # Ensure all audio finishes playing
            if hasattr(self, 'playing') and (self.playing or self.audio_queue):
                self.wait_for_playback_completion()
        
        return metrics

async def run_immediate_playback_test():
    """Run the immediate playback test based on test case 5"""
    
    print("🎵 TTS Streaming Test Client - Immediate Playback")
    print("=" * 60)
    
    # Test case 5 from original file
    test_case = {
        "name": "Realistic Paragraph with Immediate Playback",
        "text": "The quick brown fox jumps over the lazy dog. This pangram is commonly used for testing purposes. It contains every letter of the alphabet at least once. Typography enthusiasts often use it to showcase different typefaces and fonts.",
        "expected_benefit": "Immediate audio playback when first chunk arrives"
    }
    
    async with TTSStreamingClientWithPlayback() as client:
        # Check service health
        try:
            health = await client.check_health()
            print("✅ Service Health Check:")
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   Pipeline Ready: {health.get('pipeline_ready', False)}")
            print(f"   G2P Available: {health.get('g2p_service_available', False)}")
            
        except Exception as e:
            print(f"❌ Service not available: {e}")
            print("Please start the TTS service first:")
            print("   cd kokoro/kokoro_server")
            print("   python tts_service.py")
            return
        
        print("\n" + "=" * 60)
        print(f"\n🧪 Test: {test_case['name']}")
        print(f"Text ({len(test_case['text'])} chars): {test_case['text'][:80]}{'...' if len(test_case['text']) > 80 else ''}")
        print(f"Expected: {test_case['expected_benefit']}")
        
        # Test streaming with immediate playback
        print("\n🎵 Starting streaming with immediate playback...")
        try:
            # Create filename for saved audio
            audio_filename = "test_audio_immediate_playback.wav"
            
            metrics = await client.stream_tts_with_immediate_playback(
                test_case['text'], 
                save_file=audio_filename,
                min_chunk_size=4096  # Minimum chunk size before starting playback
            )
            
            if metrics['success']:
                print(f"\n✅ Streaming with Immediate Playback SUCCESS:")
                print(f"   Time to first byte: {metrics['time_to_first_byte']:.3f}s")
                print(f"   Time to first playback: {metrics['time_to_first_playback']:.3f}s")
                print(f"   Playback delay: {metrics['time_to_first_playback'] - metrics['time_to_first_byte']:.3f}s")
                print(f"   Total time: {metrics['total_time']:.3f}s")
                print(f"   Chunks received: {metrics['chunks_received']}")
                print(f"   Chunks played: {metrics['chunks_played']}")
                print(f"   Total bytes: {metrics['total_bytes']:,}")
                if metrics.get('saved_file'):
                    print(f"   Complete audio saved: {metrics['saved_file']}")
                
                # Calculate performance metrics
                if metrics['time_to_first_playback'] > 0:
                    chars_per_sec_to_playback = len(test_case['text']) / metrics['time_to_first_playback']
                    print(f"   Speed to first playback: {chars_per_sec_to_playback:.1f} chars/sec")
                
                print(f"\n🎯 Immediate Playback Benefits:")
                print(f"   ✅ Audio starts playing {metrics['time_to_first_playback']:.3f}s after request")
                print(f"   ✅ User hears audio while remaining chunks are still streaming")
                print(f"   ✅ Significantly improved perceived latency")
                
                # Wait for all audio chunks to finish playing
                print(f"\n⏳ Ensuring all audio chunks are played...")
                # The wait_for_playback_completion is called in the finally block of stream_tts_with_immediate_playback
                await asyncio.sleep(2)  # Brief pause to let the completion message show
                
            else:
                print(f"\n❌ Streaming FAILED: {metrics.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
        
        print("\n" + "=" * 60)
        print("🎵 Immediate Playback Test Complete!")
        print("This test demonstrates how streaming TTS with immediate playback")
        print("significantly improves user experience by starting audio playback")
        print("as soon as the first audio chunk is received, rather than waiting")
        print("for the complete audio generation to finish.")

if __name__ == "__main__":
    try:
        asyncio.run(run_immediate_playback_test())
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)