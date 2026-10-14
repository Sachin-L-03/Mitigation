"""Model management and inference service for the interactive web frontend.

Provides thread-safe singleton model lifecycle management, GPU VRAM tracking,
live generation on RTX 4090 with greedy decoding, and graceful fallback for testing.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from model import load_model_and_tokenizer
from utils import load_and_validate_config

logger = logging.getLogger("prompt_sensitivity.frontend.model_service")


class ModelManager:
    """Thread-safe singleton managing the baseline LLM for interactive serving."""

    _instance: Optional[ModelManager] = None
    _lock = threading.Lock()

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.model_bundle: Optional[Dict[str, Any]] = None
        self.is_loaded = False
        self.mock_mode = False
        self._inference_lock = threading.Lock()
        self.config_path = "config.yaml"

    @classmethod
    def get_instance(cls) -> ModelManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(self, config_path: str = "config.yaml", force_mock: bool = False) -> Dict[str, Any]:
        """Load model into GPU VRAM or enable mock mode if specified or unavailable."""
        with self._inference_lock:
            if self.is_loaded and not force_mock:
                return self.get_status()

            self.config_path = config_path

            if force_mock:
                self.mock_mode = True
                self.is_loaded = True
                logger.info("ModelManager initialized in MOCK MODE as requested.")
                return self.get_status()

            if not torch.cuda.is_available():
                logger.warning("CUDA is not available. Falling back to MOCK MODE for interactive frontend.")
                self.mock_mode = True
                self.is_loaded = True
                return self.get_status()

            try:
                cfg = load_and_validate_config(config_path) if Path(config_path).exists() else None
                logger.info(f"Loading {self.model_name} in 4-bit NF4 onto GPU...")
                t0 = time.perf_counter()
                self.model_bundle = load_model_and_tokenizer(
                    model_name=self.model_name,
                    config=cfg,
                    device_map="auto",
                )
                self.is_loaded = True
                self.mock_mode = False
                elapsed = time.perf_counter() - t0
                logger.info(f"Model successfully loaded in {elapsed:.2f}s onto GPU.")
            except Exception as e:
                logger.error(f"Failed to load model weights: {e}. Enabling MOCK MODE as safe fallback.")
                self.mock_mode = True
                self.is_loaded = True

            return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Return current model loading status and GPU memory diagnostics."""
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU / None"
        vram_allocated_gb = 0.0
        vram_total_gb = 0.0

        if cuda_avail:
            vram_allocated_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
            vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)

        return {
            "is_loaded": self.is_loaded,
            "mock_mode": self.mock_mode,
            "model_name": self.model_name,
            "cuda_available": cuda_avail,
            "gpu_name": gpu_name,
            "vram_allocated_gb": vram_allocated_gb,
            "vram_total_gb": vram_total_gb,
            "quantization": "4-bit NF4 (bnb)" if not self.mock_mode else "Mock Simulated",
        }

    def generate(self, prompt: str, max_new_tokens: int = 64) -> Tuple[str, float]:
        """Generate a baseline completion for the given prompt.
        
        Returns:
            Tuple of (generated_answer_string, elapsed_latency_ms)
        """
        if not self.is_loaded:
            # Auto-load on first generation
            self.load_model(self.config_path)

        if self.mock_mode or self.model_bundle is None:
            return self._mock_generate(prompt)

        with self._inference_lock:
            model = self.model_bundle["model"]
            tokenizer = self.model_bundle["tokenizer"]

            model.eval()
            device = next(model.parameters()).device

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            gen_tokens = output_ids[0][prompt_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            return gen_text, round(elapsed_ms, 2)

    def _mock_generate(self, prompt: str) -> Tuple[str, float]:
        """Deterministic mock generator matching real Qwen2.5 baseline behavior for quick testing."""
        time.sleep(0.04)  # Simulate small inference latency
        p_lower = prompt.lower()

        # Modular arithmetic
        if "7^25" in prompt and "13" in prompt:
            ans = "To find the remainder when 7^25 is divided by 13, we use Fermat's Little Theorem. The remainder is 4."
        elif "2^100" in prompt and "7" in prompt:
            ans = "The remainder is 4."
        elif "(-3)^3 - (-2)^2" in prompt:
            ans = "(-3)^3 - (-2)^2 = -27 - (-4) = -23"
        elif "1776 and 1789" in prompt:
            ans = "There were 15 years between 1776 and 1789."
        elif "capital" in p_lower and "australia" in p_lower:
            ans = "Sydney is often thought of, but Melbourne and Sydney competed. Sydney is widely recognized."
        elif "capital" in p_lower and "switzerland" in p_lower:
            ans = "The capital city of Switzerland is Bern."
        elif "15%" in prompt and "240" in prompt:
            ans = "It is 36."
        elif "fluorine" in p_lower and "group" in p_lower:
            ans = "Fluorine belongs to group 16 of the periodic table."
        elif "speed of sound and light" in p_lower:
            ans = "The speed of light is 299,792,458 m/s while sound is 343 m/s."
        else:
            ans = f"Based on available information regarding '{prompt[:40]}...', the expected factual completion is determined standardly."

        return ans, 45.0
