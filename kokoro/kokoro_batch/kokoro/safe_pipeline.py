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
import json

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
        cache_dir: Optional[str]
    ):
        """
        Initialize safe pipeline.
        
        Args:
            repo_id: HuggingFace model repository
            model: Pre-initialized KModel instance
            device: Device to use ('cuda', 'cpu', 'mps', or None for auto)
        """
        self.repo_id = 'hexgrad/Kokoro-82M'
        self.config = None
        self.kokoro = None
        self.voices = {}
        self.cache_dir = cache_dir
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        logger.info(f"Using device: {self.device}")

        self._preload_model()

        self.model = KModel(config=self.config, model=self.kokoro).to(self.device).eval()
        
    def _preload_model(self):

        logger.info(f"Loading config from repo: {self.repo_id}")
        if not isinstance(self.config, dict):
            if not self.config:
                logger.debug("No config provided, downloading from HF")
                self.config = hf_hub_download(repo_id=self.repo_id, filename='config.json', cache_dir=self.cache_dir)
            with open(self.config, 'r', encoding='utf-8') as r:
                self.config = json.load(r)
                logger.debug(f"Loaded config: {self.config}")

        logger.info(f"Loading model from repo: {self.repo_id}")
        if not self.kokoro:
            self.kokoro = hf_hub_download(repo_id=self.repo_id, filename=KModel.MODEL_NAMES[self.repo_id], cache_dir=self.cache_dir)

        logger.info("Loading all available voices...")
        try:
            # Get a list of all files in the 'voices' directory of the repo
            repo_files = list_repo_files(repo_id=self.repo_id, repo_type='model')
            voice_files = [f for f in repo_files if f.startswith('voices/') and f.endswith('.pt')]
            
            for voice_file in voice_files:
                voice_name = voice_file.split('/')[-1].replace('.pt', '')
                if voice_name not in self.voices:
                    try:
                        # This will download the file and cache it locally
                        voice_path = hf_hub_download(repo_id=self.repo_id, filename=voice_file, cache_dir=self.cache_dir)

                        # Load the tensor and cache it in memory
                        voice_tensor = torch.load(voice_path, map_location=self.device, weights_only=True)
                        self.voices[voice_name] = voice_tensor

                        # We don't need to load the tensor here, just ensure it's downloaded
                        logger.debug(f"Successfully preloaded voice: {voice_name}")
                    except Exception as e:
                        logger.warning(f"Failed to preload voice {voice_name}: {e}")
            
            logger.info(f"✅ Preloaded {len(voice_files)} voices.")
            
        except Exception as e:
            logger.error(f"❌ Could not preload voices: {e}")
    
    def from_phonemes(
        self, 
        phonemes: list[str], 
        voices: list[str],
        speeds: list[float]
    ) -> torch.FloatTensor:
        if len(phonemes) > 510:
            raise ValueError(f'Phoneme string too long: {len(phonemes)} > 510')

        voice_packs = [self.voices[voice].to(self.model.device) for voice in voices]

        # Use the model directly with phonemes
        # The voice pack needs to be indexed by phoneme length - 1
        voice_tensor = []
        for v, p in zip(voice_packs, phonemes):
            voice_tensor.append(v[len(p)-1])
        output = self.model(phonemes, voice_tensor, speeds, return_output=True)
        return output.audio
    
        '''
        device = self.model.device
        voice_packs = torch.stack([self.voices[voice].to(device) for voice in voices])  # shape [batch, 510, 1, 256]

        phoneme_lengths = torch.tensor([len(p) for p in phonemes], device=device) - 1  # shape [batch]

        # Advanced indexing: select voice embeddings by phoneme length for each sample in batch
        # voice_packs shape: [batch, 510, 1, 256]
        # phoneme_lengths shape: [batch]
        # gather embeddings at phoneme_lengths indices along dim=1
        indices = phoneme_lengths.view(-1, 1, 1).expand(-1, 1, voice_packs.size(-1))
        selected_voice_tensors = torch.gather(voice_packs, dim=1, index=indices).squeeze(1)  # shape [batch, 1, 256]
        '''