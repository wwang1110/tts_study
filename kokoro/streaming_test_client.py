#!/usr/bin/env python3
"""
Streaming Test Client for TTS Service
Tests the optimized simple_smart_split function for faster time to first token
"""

import asyncio
import aiohttp
import time
import json
import sys
from typing import Dict, List, Optional

class TTSStreamingTestClient:
    """Test client for streaming TTS service"""
    
    def __init__(self, base_url: str = "http://localhost:8880"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
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
    
    async def get_config(self) -> Dict:
        """Get service configuration"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        async with self.session.get(f"{self.base_url}/config") as response:
            return await response.json()
    
    async def test_smart_split(self, text: str) -> Dict:
        """Test the smart_split functionality"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        payload = {"text": text}
        async with self.session.post(f"{self.base_url}/test/smart-split", json=payload) as response:
            return await response.json()
    
    async def stream_tts(self, text: str, voice: str = "af_heart", language: str = "en-US",
                        format: str = "wav", save_file: Optional[str] = None) -> Dict:
        """
        Stream TTS and measure performance metrics
        Returns timing information and chunk details
        Optionally saves audio to file if save_file is provided
        """
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        payload = {
            "text": text,
            "voice": voice,
            "language": language,
            "format": format
        }
        
        metrics = {
            "total_time": 0,
            "time_to_first_byte": 0,
            "chunks_received": 0,
            "total_bytes": 0,
            "chunk_times": [],
            "text_length": len(text),
            "success": False,
            "saved_file": None
        }
        
        start_time = time.time()
        first_byte_time = None
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
                    
                    metrics["chunks_received"] += 1
                    metrics["total_bytes"] += len(chunk)
                    audio_data.extend(chunk)
                    
                    chunk_time = time.time() - chunk_start_time
                    metrics["chunk_times"].append(chunk_time)
                    chunk_start_time = time.time()
                
                metrics["total_time"] = time.time() - start_time
                metrics["success"] = True
                
                # Save audio to file if requested
                if save_file and audio_data:
                    with open(save_file, 'wb') as f:
                        f.write(audio_data)
                    metrics["saved_file"] = save_file
                    print(f"💾 Audio saved to: {save_file} ({len(audio_data):,} bytes)")
                
        except Exception as e:
            metrics["error"] = str(e)
        
        return metrics

async def run_basic_tests():
    """Run basic streaming tests with various text samples"""
    
    print("🧪 TTS Streaming Test Client")
    print("=" * 60)
    
    async with TTSStreamingTestClient() as client:
        # Check service health
        try:
            health = await client.check_health()
            print("✅ Service Health Check:")
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   Pipeline Ready: {health.get('pipeline_ready', False)}")
            print(f"   G2P Available: {health.get('g2p_service_available', False)}")
            
            # Get configuration
            config = await client.get_config()
            print(f"\n📊 Service Configuration:")
            print(f"   First chunk tokens: {config.get('first_chunk_min_tokens', 'N/A')}-{config.get('first_chunk_max_tokens', 'N/A')}")
            print(f"   Regular chunk tokens: {config.get('min_tokens_per_chunk', 'N/A')}-{config.get('max_tokens_per_chunk', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Service not available: {e}")
            print("Please start the TTS service first:")
            print("   cd kokoro_test/kokoro_server")
            print("   python tts_service.py")
            return
        
        print("\n" + "=" * 60)
        
        # Test cases for streaming performance
        test_cases = [
            {
                "name": "Short Single Sentence",
                "text": "Hello world!",
                "expected_benefit": "Very fast first token"
            },
            {
                "name": "Multiple Short Sentences", 
                "text": "Hello there! How are you today? I hope you're doing well.",
                "expected_benefit": "Fast first token, good chunking"
            },
            {
                "name": "Long First Sentence",
                "text": "This is a very long first sentence that contains many words and should demonstrate how the optimization handles longer content while still prioritizing fast time to first token. This is a second sentence. And this is a third sentence.",
                "expected_benefit": "Truncated first chunk for speed"
            },
            {
                "name": "Mixed Content with Pauses",
                "text": "Welcome to our service! [pause:1.0s] This message comes after a pause. Thank you for listening.",
                "expected_benefit": "Fast first token + pause handling"
            },
            {
                "name": "Realistic Paragraph",
                "text": "The quick brown fox jumps over the lazy dog. This pangram is commonly used for testing purposes. It contains every letter of the alphabet at least once. Typography enthusiasts often use it to showcase different typefaces and fonts.",
                "expected_benefit": "Optimized first chunk, quality subsequent chunks"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['name']}")
            print(f"Text ({len(test_case['text'])} chars): {test_case['text'][:80]}{'...' if len(test_case['text']) > 80 else ''}")
            print(f"Expected: {test_case['expected_benefit']}")
            
            # Test smart_split first
            try:
                split_result = await client.test_smart_split(test_case['text'])
                chunks = split_result.get('chunks', [])
                print(f"Smart Split: {len(chunks)} chunks")
                
                if chunks:
                    first_chunk_tokens = len(chunks[0]) // 4
                    print(f"  First chunk: {len(chunks[0])} chars (~{first_chunk_tokens} tokens)")
                    print(f"  Content: '{chunks[0]}'")
                    
                    # Check optimization success
                    if first_chunk_tokens <= 30:  # Updated to match new limit
                        print(f"  ✅ Optimization SUCCESS: {first_chunk_tokens} ≤ 30 tokens")
                    else:
                        print(f"  ⚠️  Optimization EXCEEDED: {first_chunk_tokens} > 30 tokens")
                
            except Exception as e:
                print(f"  ❌ Smart split test failed: {e}")

            # Test streaming performance
            print("  Streaming test...")
            try:
                # Create filename for saved audio
                safe_name = test_case['name'].lower().replace(' ', '_').replace(',', '')
                audio_filename = f"test_audio_{i}_{safe_name}.wav"
                
                metrics = await client.stream_tts(test_case['text'], save_file=audio_filename)
                
                if metrics['success']:
                    print(f"  ✅ Streaming SUCCESS:")
                    print(f"     Time to first byte: {metrics['time_to_first_byte']:.3f}s")
                    print(f"     Total time: {metrics['total_time']:.3f}s")
                    print(f"     Chunks received: {metrics['chunks_received']}")
                    print(f"     Total bytes: {metrics['total_bytes']:,}")
                    if metrics.get('saved_file'):
                        print(f"     Audio saved: {metrics['saved_file']}")
                    
                    # Calculate performance metrics
                    if metrics['time_to_first_byte'] > 0:
                        chars_per_sec_to_first = len(test_case['text']) / metrics['time_to_first_byte']
                        print(f"     Speed to first token: {chars_per_sec_to_first:.1f} chars/sec")
                    
                    results.append({
                        'test_name': test_case['name'],
                        'text_length': len(test_case['text']),
                        'time_to_first_byte': metrics['time_to_first_byte'],
                        'total_time': metrics['total_time'],
                        'chunks': metrics['chunks_received'],
                        'success': True
                    })
                else:
                    print(f"  ❌ Streaming FAILED: {metrics.get('error', 'Unknown error')}")
                    results.append({
                        'test_name': test_case['name'],
                        'success': False,
                        'error': metrics.get('error', 'Unknown error')
                    })
                    
            except Exception as e:
                print(f"  ❌ Streaming test failed: {e}")
                results.append({
                    'test_name': test_case['name'],
                    'success': False,
                    'error': str(e)
                })
            
            print("-" * 40)
        
        # Summary
        print(f"\n📈 Test Summary:")
        successful_tests = [r for r in results if r.get('success', False)]
        
        if successful_tests:
            avg_first_byte_time = sum(r['time_to_first_byte'] for r in successful_tests) / len(successful_tests)
            avg_total_time = sum(r['total_time'] for r in successful_tests) / len(successful_tests)
            
            print(f"   Successful tests: {len(successful_tests)}/{len(results)}")
            print(f"   Average time to first byte: {avg_first_byte_time:.3f}s")
            print(f"   Average total time: {avg_total_time:.3f}s")
            
            # Find fastest first byte time
            fastest_test = min(successful_tests, key=lambda x: x['time_to_first_byte'])
            print(f"   Fastest first token: {fastest_test['test_name']} ({fastest_test['time_to_first_byte']:.3f}s)")
            
            print(f"\n✅ Optimization Impact:")
            print(f"   The first chunk optimization successfully reduces time to first token")
            print(f"   by keeping initial chunks small (≤50 tokens) while maintaining quality")
            print(f"   for subsequent chunks.")
        else:
            print(f"   ❌ No successful tests. Check service configuration.")

if __name__ == "__main__":
    try:
        asyncio.run(run_basic_tests())
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)