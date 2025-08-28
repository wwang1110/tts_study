#!/usr/bin/env python3
"""
License-Safe Kokoro Pipeline

This pipeline avoids GPL dependencies by:
1. Using direct phoneme input (recommended)
2. Optionally using external G2P service (GPL isolated)

Safe for production use with commercial licenses.
"""

from kokoro.model import KModel
from huggingface_hub import hf_hub_download, list_repo_files
from typing import Optional, Union, Generator
import torch
import logging

logger = logging.getLogger(__name__)

class SafePipeline:
    """
    License-safe TTS pipeline that avoids GPL dependencies.
    
    Usage patterns:
    1. Direct phonemes: pipeline.from_phonemes("həˈloʊ", "af_heart")
    2. G2P service: pipeline.from_text("hello", "af_heart", g2p_url="http://0.0.0.0:5000")
    """
    
    def __init__(
        self,
        repo_id: str = 'hexgrad/Kokoro-82M',
        model: Optional[KModel] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = './.cache'
    ):
        """
        Initialize safe pipeline.
        
        Args:
            repo_id: HuggingFace model repository
            model: Pre-initialized KModel instance
            device: Device to use ('cuda', 'cpu', 'mps', or None for auto)
        """
        self.repo_id = repo_id
        self.voices = {}
        self.cache_dir = cache_dir
        
        # Initialize model
        if isinstance(model, KModel):
            self.model = model
        else:
            if device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            if device == 'mps' and not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            if device is None:
                if torch.cuda.is_available():
                    device = 'cuda'
                else:
                    device = 'cpu'
            
            self.model = KModel(repo_id=repo_id, cache_dir=cache_dir).to(device).eval()
        
        # Preload all voices
        self._preload_voices()
    
    def _preload_voices(self):
        """Download and cache all available voices from the HuggingFace Hub."""
        logger.info("Preloading all available voices...")
        try:
            # Get a list of all files in the 'voices' directory of the repo
            repo_files = list_repo_files(repo_id=self.repo_id, repo_type='model')
            voice_files = [f for f in repo_files if f.startswith('voices/') and f.endswith('.pt')]
            
            for voice_file in voice_files:
                voice_name = voice_file.split('/')[-1].replace('.pt', '')
                if voice_name not in self.voices:
                    try:
                        # This will download the file and cache it locally
                        hf_hub_download(
                            repo_id=self.repo_id,
                            filename=voice_file,
                            cache_dir=self.cache_dir
                        )
                        # We don't need to load the tensor here, just ensure it's downloaded
                        logger.debug(f"Successfully preloaded voice: {voice_name}")
                    except Exception as e:
                        logger.warning(f"Failed to preload voice {voice_name}: {e}")
            
            logger.info(f"✅ Preloaded {len(voice_files)} voices.")
            
        except Exception as e:
            logger.error(f"❌ Could not preload voices: {e}")
    def load_voice(self, voice: Union[str, torch.FloatTensor]) -> torch.FloatTensor:
        """
        Load voice embedding from file or return existing tensor.
        
        Args:
            voice: Voice name (e.g., 'af_heart') or voice tensor
            
        Returns:
            Voice embedding tensor
        """
        if isinstance(voice, torch.FloatTensor):
            return voice
            
        if voice in self.voices:
            return self.voices[voice]
        
        logger.debug(f"Loading voice: {voice}")
        
        if voice.endswith('.pt'):
            voice_path = voice
        else:
            voice_path = hf_hub_download(
                repo_id=self.repo_id, 
                filename=f'voices/{voice}.pt',
                cache_dir=self.cache_dir
            )
        
        # Load the tensor and cache it in memory
        voice_tensor = torch.load(voice_path, map_location=self.model.device, weights_only=True)
        self.voices[voice] = voice_tensor
        
        logger.debug(f"Loaded and cached voice '{voice}' in memory.")
        return voice_tensor
    
    def from_phonemes(
        self, 
        phonemes: str, 
        voice: Union[str, torch.FloatTensor],
        speed: float = 1.0
    ) -> torch.FloatTensor:
        """
        Generate audio directly from phonemes (no GPL dependencies).
        
        Args:
            phonemes: Phoneme string (e.g., "həˈloʊ wɜrld")
            voice: Voice name or tensor
            speed: Speech speed multiplier
            
        Returns:
            Audio tensor
        """
        if len(phonemes) > 510:
            raise ValueError(f'Phoneme string too long: {len(phonemes)} > 510')
        
        voice_pack = self.load_voice(voice).to(self.model.device)
        
        # Use the model directly with phonemes
        # The voice pack needs to be indexed by phoneme length - 1
        voice_tensor = voice_pack[len(phonemes)-1]
        output = self.model(phonemes, voice_tensor, speed, return_output=True)
        return output.audio
    
    def from_text(
        self,
        text: str,
        voice: Union[str, torch.FloatTensor],
        lang: str = "en-US",
        g2p_url: str = "http://0.0.0.0:5000",
        speed: float = 1.0
    ) -> torch.FloatTensor:
        """
        Generate audio from text using external G2P service.
        
        Args:
            text: Input text
            voice: Voice name or tensor
            lang: Language code (en-US, en-GB, pt-BR, fr-FR, es, hi, it, ja, zh)
            g2p_url: URL of G2P service
            speed: Speech speed multiplier
            
        Returns:
            Audio tensor
            
        Raises:
            ImportError: If requests is not available
            RuntimeError: If G2P service is unavailable
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required for G2P service. Install with: pip install requests")
        
        try:
            response = requests.post(
                f"{g2p_url}/convert",
                json={"text": text, "lang": lang},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            if 'error' in data:
                raise RuntimeError(f"G2P service error: {data['error']}")
            
            phonemes = data['phonemes']
            return self.from_phonemes(phonemes, voice, speed)
            
        except requests.RequestException as e:
            raise RuntimeError(f"G2P service unavailable: {e}")
    
    def batch_from_phonemes(
        self,
        phoneme_list: list[str],
        voice: Union[str, torch.FloatTensor],
        speed: float = 1.0
    ) -> Generator[torch.FloatTensor, None, None]:
        """
        Generate audio from multiple phoneme strings.
        
        Args:
            phoneme_list: List of phoneme strings
            voice: Voice name or tensor
            speed: Speech speed multiplier
            
        Yields:
            Audio tensors
        """
        for phonemes in phoneme_list:
            yield self.from_phonemes(phonemes, voice, speed)

# Convenience functions for backward compatibility
def create_safe_pipeline(
    repo_id: str = 'hexgrad/Kokoro-82M',
    device: Optional[str] = None
) -> SafePipeline:
    """Create a license-safe pipeline instance."""
    return SafePipeline(repo_id=repo_id, device=device)

def phonemes_to_audio(
    phonemes: str,
    voice: Union[str, torch.FloatTensor],
    repo_id: str = 'hexgrad/Kokoro-82M',
    device: Optional[str] = None,
    speed: float = 1.0
) -> torch.FloatTensor:
    """
    Simple function to convert phonemes to audio.
    
    Args:
        phonemes: Phoneme string
        voice: Voice name or tensor
        repo_id: Model repository
        device: Device to use
        speed: Speech speed
        
    Returns:
        Audio tensor
    """
    pipeline = SafePipeline(repo_id=repo_id, device=device)
    return pipeline.from_phonemes(phonemes, voice, speed)