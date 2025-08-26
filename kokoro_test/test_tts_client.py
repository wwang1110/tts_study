#!/usr/bin/env python3
"""
Test client for TTS service demonstrating all three modes:
1. Single TTS
2. Batch TTS  
3. Streaming TTS
"""

import requests
import json
import time
import os
from typing import Dict, Any

# TTS Service URL
BASE_URL = "http://localhost:8880"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Service healthy: {health_data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_single_tts():
    """Test Mode 1: Single TTS"""
    print("\n🎵 Testing Single TTS...")
    
    request_data = {
        "text": "Hello world, this is a test of the single TTS endpoint.",
        "voice": "af_heart",
        "language": "en-US",
        "speed": 1.0,
        "format": "wav"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/tts",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            # Save the audio file
            filename = "test_single_tts.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            
            print(f"✅ Single TTS successful!")
            print(f"   📁 Saved: {filename}")
            print(f"   📊 Size: {len(response.content)} bytes")
            print(f"   🎤 Voice: {response.headers.get('X-Voice-Used', 'unknown')}")
            print(f"   ⏱️  Duration: {response.headers.get('X-Audio-Duration', 'unknown')}s")
            return True
        else:
            print(f"❌ Single TTS failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Single TTS error: {e}")
        return False

def test_batch_tts():
    """Test Mode 2: Batch TTS"""
    print("\n📦 Testing Batch TTS...")
    
    batch_requests = [
        {
            "text": "This is the first batch request.",
            "voice": "af_heart",
            "language": "en-US",
            "speed": 1.0,
            "format": "wav"
        },
        {
            "text": "This is the second batch request with a different voice.",
            "voice": "af_bella",
            "language": "en-US", 
            "speed": 1.2,
            "format": "wav"
        },
        {
            "text": "Hola mundo, esta es una prueba en español.",
            "voice": "af_sarah",
            "language": "es",
            "speed": 0.9,
            "format": "wav"
        }
    ]
    
    request_data = {
        "requests": batch_requests
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/tts/batch",
            json=request_data,
            timeout=60
        )
        
        if response.status_code == 200:
            # Save the ZIP file
            filename = "test_batch_tts.zip"
            with open(filename, "wb") as f:
                f.write(response.content)
            
            print(f"✅ Batch TTS successful!")
            print(f"   📁 Saved: {filename}")
            print(f"   📊 Size: {len(response.content)} bytes")
            print(f"   📈 Requests: {response.headers.get('X-Total-Requests', 'unknown')}")
            return True
        else:
            print(f"❌ Batch TTS failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch TTS error: {e}")
        return False

def test_streaming_tts():
    """Test Mode 3: Streaming TTS"""
    print("\n🌊 Testing Streaming TTS...")
    
    request_data = {
        "text": "This is a streaming TTS test. The audio should be delivered in real-time chunks as it's being generated.",
        "voice": "af_heart",
        "language": "en-US",
        "speed": 1.0,
        "format": "wav"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/tts/stream",
            json=request_data,
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            # Stream and save the audio
            filename = "test_streaming_tts.wav"
            total_bytes = 0
            chunks_received = 0
            
            print("   📡 Receiving stream...")
            start_time = time.time()
            
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)
                        chunks_received += 1
                        
                        # Show progress every 10 chunks
                        if chunks_received % 10 == 0:
                            elapsed = time.time() - start_time
                            print(f"   📊 Received {chunks_received} chunks, {total_bytes} bytes in {elapsed:.2f}s")
            
            elapsed = time.time() - start_time
            print(f"✅ Streaming TTS successful!")
            print(f"   📁 Saved: {filename}")
            print(f"   📊 Total: {total_bytes} bytes in {chunks_received} chunks")
            print(f"   ⏱️  Time: {elapsed:.2f}s")
            print(f"   🚀 Rate: {total_bytes/elapsed:.0f} bytes/sec")
            return True
        else:
            print(f"❌ Streaming TTS failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Streaming TTS error: {e}")
        return False

def test_list_voices():
    """Test the voices endpoint"""
    print("\n🎤 Testing List Voices...")
    
    try:
        response = requests.get(f"{BASE_URL}/voices")
        if response.status_code == 200:
            voices_data = response.json()
            voices = voices_data.get("voices", [])
            print(f"✅ Found {len(voices)} voices:")
            for voice in voices[:5]:  # Show first 5
                print(f"   🎵 {voice}")
            if len(voices) > 5:
                print(f"   ... and {len(voices) - 5} more")
            return True
        else:
            print(f"❌ List voices failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ List voices error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TTS Service Test Client")
    print("=" * 50)
    
    # Test health first
    if not test_health_check():
        print("\n❌ Service not healthy, stopping tests")
        return
    
    # Run all tests
    tests = [
        ("List Voices", test_list_voices),
        ("Single TTS", test_single_tts),
        ("Batch TTS", test_batch_tts),
        ("Streaming TTS", test_streaming_tts)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! TTS service is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the service and G2P service status.")

if __name__ == "__main__":
    main()