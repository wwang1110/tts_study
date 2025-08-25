#!/usr/bin/env python3
"""
License-Safe G2P Client

This client communicates with the G2P service without importing any GPL dependencies.
Safe for use in production applications with commercial licenses.

Usage:
    from g2p_client import text_to_phonemes
    phonemes = text_to_phonemes("Hello world")
"""

import requests
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def text_to_phonemes(
    text: str, 
    service_url: str = "http://localhost:5000",
    lang: str = "en",
    timeout: int = 30
) -> str:
    """
    Convert text to phonemes using external G2P service.
    
    Args:
        text: Input text to convert
        service_url: URL of the G2P service
        lang: Language code (default: "en")
        timeout: Request timeout in seconds
        
    Returns:
        Phonemes string
        
    Raises:
        requests.RequestException: If service is unavailable
        ValueError: If conversion fails
    """
    try:
        response = requests.post(
            f"{service_url}/convert",
            json={"text": text, "lang": lang},
            timeout=timeout
        )
        response.raise_for_status()
        
        data = response.json()
        if 'error' in data:
            raise ValueError(f"G2P service error: {data['error']}")
            
        return data['phonemes']
        
    except requests.RequestException as e:
        logger.error(f"G2P service request failed: {e}")
        raise
    except (KeyError, ValueError) as e:
        logger.error(f"G2P service response error: {e}")
        raise ValueError(f"Invalid response from G2P service: {e}")

def check_service_health(service_url: str = "http://localhost:5000") -> bool:
    """
    Check if G2P service is available.
    
    Args:
        service_url: URL of the G2P service
        
    Returns:
        True if service is healthy, False otherwise
    """
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

class G2PClient:
    """
    G2P client class for more advanced usage patterns.
    """
    
    def __init__(self, service_url: str = "http://localhost:5000"):
        self.service_url = service_url
        
    def convert(self, text: str, lang: str = "en") -> str:
        """Convert text to phonemes."""
        return text_to_phonemes(text, self.service_url, lang)
        
    def is_available(self) -> bool:
        """Check if service is available."""
        return check_service_health(self.service_url)
        
    def batch_convert(self, texts: list[str], lang: str = "en") -> list[str]:
        """Convert multiple texts to phonemes."""
        return [self.convert(text, lang) for text in texts]

if __name__ == "__main__":
    # Example usage
    try:
        # Check service health
        if not check_service_health():
            print("G2P service is not available. Start it with: python g2p_service.py")
            exit(1)
            
        # Convert text to phonemes
        text = "Hello world, this is a test."
        phonemes = text_to_phonemes(text)
        print(f"Text: {text}")
        print(f"Phonemes: {phonemes}")
        
    except Exception as e:
        print(f"Error: {e}")