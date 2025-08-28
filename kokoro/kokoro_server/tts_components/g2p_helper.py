from typing import Optional

async def text_to_phonemes(
    text: str,
    language: str,
    g2p_session,
    g2p_url: str,
    g2p_timeout: int
) -> str:
    """Convert text to phonemes with aiohttp fallback to requests"""
    
    # Try aiohttp first (fastest)
    if g2p_session:
        try:
            async with g2p_session.post(
                f"{g2p_url}/convert",
                json={"text": text, "lang": language}
            ) as response:
                response.raise_for_status()
                
                data = await response.json()
                if 'error' in data:
                    raise RuntimeError(f"G2P service error: {data['error']}")
                
                return data['phonemes']
        except Exception as e:
            raise RuntimeError(f"G2P service unavailable at {g2p_url}: {e}")
    
    # Fallback to requests library (slower but works)
    else:
        try:
            import requests
            
            response = requests.post(
                f"{g2p_url}/convert",
                json={"text": text, "lang": language},
                timeout=g2p_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            if 'error' in data:
                raise RuntimeError(f"G2P service error: {data['error']}")
            
            return data['phonemes']
            
        except ImportError:
            raise RuntimeError("Neither aiohttp nor requests library available for G2P service")
        except Exception as e:
            raise RuntimeError(f"G2P service unavailable at {g2p_url}: {e}")