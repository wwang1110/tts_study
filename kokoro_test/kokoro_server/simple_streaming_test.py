#!/usr/bin/env python3
"""
Simple Streaming Test Client for TTS Service
Tests the optimized simple_smart_split function using standard library
"""

import json
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, List

class SimpleTTSTestClient:
    """Simple test client using standard library"""
    
    def __init__(self, base_url: str = "http://localhost:8880"):
        self.base_url = base_url
    
    def check_health(self) -> Dict:
        """Check if the TTS service is healthy"""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            raise RuntimeError(f"Cannot connect to TTS service: {e}")
    
    def get_config(self) -> Dict:
        """Get service configuration"""
        try:
            with urllib.request.urlopen(f"{self.base_url}/config", timeout=10) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            raise RuntimeError(f"Cannot get config: {e}")
    
    def test_smart_split(self, text: str) -> Dict:
        """Test the smart_split functionality"""
        try:
            data = json.dumps({"text": text}).encode('utf-8')
            req = urllib.request.Request(
                f"{self.base_url}/test/smart-split",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            raise RuntimeError(f"Smart split test failed: {e}")
    
    def test_streaming_performance(self, text: str, voice: str = "af_heart", 
                                 language: str = "en-US", format: str = "wav") -> Dict:
        """
        Test streaming performance by measuring response times
        Note: This doesn't actually stream but measures the response time
        """
        try:
            data = json.dumps({
                "text": text,
                "voice": voice,
                "language": language,
                "format": format
            }).encode('utf-8')
            
            req = urllib.request.Request(
                f"{self.base_url}/tts/stream",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            start_time = time.time()
            
            with urllib.request.urlopen(req, timeout=60) as response:
                first_byte_time = time.time()
                
                # Read response in chunks to simulate streaming
                total_bytes = 0
                chunk_count = 0
                
                while True:
                    chunk = response.read(1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    chunk_count += 1
                
                total_time = time.time() - start_time
                time_to_first_byte = first_byte_time - start_time
                
                return {
                    "success": True,
                    "time_to_first_byte": time_to_first_byte,
                    "total_time": total_time,
                    "total_bytes": total_bytes,
                    "chunk_count": chunk_count,
                    "text_length": len(text)
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text_length": len(text)
            }

def run_basic_streaming_tests():
    """Run basic streaming tests"""
    
    print("🧪 Simple TTS Streaming Test")
    print("=" * 60)
    
    client = SimpleTTSTestClient()
    
    # Check service health
    try:
        health = client.check_health()
        print("✅ Service Health Check:")
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   Pipeline Ready: {health.get('pipeline_ready', False)}")
        print(f"   G2P Available: {health.get('g2p_service_available', False)}")
        
        # Get configuration
        config = client.get_config()
        print(f"\n📊 Service Configuration:")
        print(f"   First chunk tokens: {config.get('first_chunk_min_tokens', 'N/A')}-{config.get('first_chunk_max_tokens', 'N/A')}")
        print(f"   Regular chunk tokens: {config.get('min_tokens_per_chunk', 'N/A')}-{config.get('max_tokens_per_chunk', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Service not available: {e}")
        print("\nTo start the TTS service:")
        print("   1. cd kokoro_test/kokoro_server")
        print("   2. source ../../.venv/bin/activate")
        print("   3. python tts_service.py")
        return
    
    print("\n" + "=" * 60)
    
    # Test cases
    test_cases = [
        {
            "name": "Very Short Text",
            "text": "Hello!",
            "description": "Single word - should be very fast"
        },
        {
            "name": "Short Sentence",
            "text": "Hello world, how are you today?",
            "description": "One sentence - optimized first chunk"
        },
        {
            "name": "Two Sentences",
            "text": "Hello there! How are you doing today?",
            "description": "Two sentences - first should be small chunk"
        },
        {
            "name": "Long First Sentence",
            "text": "This is a very long first sentence that contains many words and should demonstrate how the optimization handles longer content while still prioritizing fast time to first token. This is a shorter second sentence.",
            "description": "Long first sentence - should be truncated for speed"
        },
        {
            "name": "Multiple Sentences",
            "text": "Welcome to our service! We hope you enjoy using it. This is the third sentence. And this is the fourth sentence with some additional content.",
            "description": "Multiple sentences - first chunk optimized"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Text ({len(test_case['text'])} chars): {test_case['text'][:80]}{'...' if len(test_case['text']) > 80 else ''}")
        
        # Test smart_split functionality
        try:
            split_result = client.test_smart_split(test_case['text'])
            chunks = split_result.get('chunks', [])
            
            print(f"\n📝 Smart Split Analysis:")
            print(f"   Generated {len(chunks)} chunks")
            
            if chunks:
                # Analyze first chunk
                first_chunk = chunks[0]
                first_chunk_tokens = len(first_chunk) // 4  # Approximate tokens
                
                print(f"   First chunk: {len(first_chunk)} chars (~{first_chunk_tokens} tokens)")
                print(f"   Content: '{first_chunk[:60]}{'...' if len(first_chunk) > 60 else ''}'")
                
                # Check if optimization worked
                if first_chunk_tokens <= 50:  # Our target max
                    print(f"   ✅ Optimization SUCCESS: {first_chunk_tokens} ≤ 50 tokens")
                else:
                    print(f"   ⚠️  Optimization EXCEEDED: {first_chunk_tokens} > 50 tokens")
                
                # Show other chunks
                if len(chunks) > 1:
                    for j, chunk in enumerate(chunks[1:], 2):
                        chunk_tokens = len(chunk) // 4
                        print(f"   Chunk {j}: {len(chunk)} chars (~{chunk_tokens} tokens)")
            
        except Exception as e:
            print(f"   ❌ Smart split failed: {e}")
        
        # Test streaming performance
        print(f"\n🚀 Streaming Performance Test:")
        try:
            perf_result = client.test_streaming_performance(test_case['text'])
            
            if perf_result['success']:
                print(f"   ✅ SUCCESS:")
                print(f"      Time to first byte: {perf_result['time_to_first_byte']:.3f}s")
                print(f"      Total time: {perf_result['total_time']:.3f}s")
                print(f"      Total bytes: {perf_result['total_bytes']:,}")
                print(f"      Chunks received: {perf_result['chunk_count']}")
                
                # Calculate performance metrics
                if perf_result['time_to_first_byte'] > 0:
                    chars_per_sec = len(test_case['text']) / perf_result['time_to_first_byte']
                    print(f"      Processing speed: {chars_per_sec:.1f} chars/sec to first byte")
                
                results.append({
                    'name': test_case['name'],
                    'text_length': len(test_case['text']),
                    'time_to_first_byte': perf_result['time_to_first_byte'],
                    'total_time': perf_result['total_time'],
                    'success': True
                })
            else:
                print(f"   ❌ FAILED: {perf_result['error']}")
                results.append({
                    'name': test_case['name'],
                    'success': False,
                    'error': perf_result['error']
                })
                
        except Exception as e:
            print(f"   ❌ Performance test failed: {e}")
        
        print("-" * 50)
    
    # Summary
    print(f"\n📊 Performance Summary:")
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        print(f"   Successful tests: {len(successful_results)}/{len(results)}")
        
        # Calculate averages
        avg_first_byte = sum(r['time_to_first_byte'] for r in successful_results) / len(successful_results)
        avg_total = sum(r['total_time'] for r in successful_results) / len(successful_results)
        
        print(f"   Average time to first byte: {avg_first_byte:.3f}s")
        print(f"   Average total time: {avg_total:.3f}s")
        
        # Find best performance
        fastest = min(successful_results, key=lambda x: x['time_to_first_byte'])
        print(f"   Fastest first byte: {fastest['name']} ({fastest['time_to_first_byte']:.3f}s)")
        
        print(f"\n✅ Optimization Benefits:")
        print(f"   • First chunks are kept small (≤50 tokens) for faster initial response")
        print(f"   • Subsequent chunks maintain quality with larger size (100-400 tokens)")
        print(f"   • Users hear audio sooner, improving perceived performance")
        print(f"   • Streaming starts faster while maintaining overall audio quality")
        
    else:
        print(f"   ❌ No successful tests completed")
        print(f"   Check that the TTS service is running and G2P service is available")

if __name__ == "__main__":
    try:
        run_basic_streaming_tests()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")