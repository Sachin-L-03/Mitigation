"""Unified model loading interface supporting Qwen2.5 and LLaMA-3.1.

Enforces strict hardware constraints, configurable 4-bit quantization,
memory-efficient options, and prohibits silent CPU or model fallbacks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("prompt_sensitivity.model")

SUPPORTED_MODELS: List[str] = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def resolve_dtype(dtype_str: str):
    """Map string dtype description to torch dtype."""
    import torch
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose from {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def build_quantization_config(quant_dict: Dict[str, Any]):
    """Construct Hugging Face BitsAndBytesConfig from configuration dictionary."""
    from transformers import BitsAndBytesConfig

    load_in_4bit = quant_dict.get("load_in_4bit", True)
    if not load_in_4bit:
        return None

    compute_dtype_str = quant_dict.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = resolve_dtype(compute_dtype_str)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_dict.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_dict.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    return bnb_config


def validate_model_target(model_name: str, allowed_models: Optional[List[str]] = None) -> None:
    """Ensure requested model is among the allowed models or an existing local path."""
    allowed = allowed_models or SUPPORTED_MODELS
    is_supported = model_name in allowed
    is_local_path = Path(model_name).exists()

    if not (is_supported or is_local_path):
        raise ValueError(
            f"Requested model '{model_name}' is not supported. Supported models: {allowed}. "
            "Changing models or fallback is strictly prohibited."
        )


def check_cuda_prerequisites(enforce_cuda: bool = True) -> None:
    """Verify CUDA availability and throw strict error if unavailable."""
    import torch

    if not torch.cuda.is_available():
        msg = (
            "CUDA is unavailable in this environment, but GPU execution is required. "
            "Silent CPU fallback is prohibited by project requirements."
        )
        if enforce_cuda:
            raise RuntimeError(msg)
        else:
            logger.warning(msg)


def load_model_and_tokenizer(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    device_map: str = "auto",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Unified loader for Qwen2.5 and LLaMA-3.1 models.

    Args:
        model_name: Model identifier (e.g., 'Qwen/Qwen2.5-7B-Instruct' or 'meta-llama/Llama-3.1-8B-Instruct')
        config: Loaded project configuration dictionary
        device_map: Device mapping strategy (defaults to 'auto' for GPU distribution)
        token: Hugging Face auth token (or retrieved from env HF_TOKEN)

    Returns:
        Dictionary containing loaded 'model', 'tokenizer', 'model_name', and diagnostic metadata.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. Validate requested model
    allowed_models = (
        config.get("model", {}).get("supported_models", SUPPORTED_MODELS)
        if config
        else SUPPORTED_MODELS
    )
    validate_model_target(model_name, allowed_models)

    # 2. Strict CUDA requirement check
    enforce_cuda = config.get("hardware", {}).get("enforce_cuda", True) if config else True
    check_cuda_prerequisites(enforce_cuda=enforce_cuda)

    # Check fallback prohibition
    fallback_allowed = config.get("model", {}).get("fallback_allowed", False) if config else False
    if fallback_allowed:
        raise ValueError("model.fallback_allowed must be False. Silent fallback is prohibited.")

    # 3. Retrieve auth token if provided
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # 4. Resolve precision & quantization
    torch_dtype_str = (
        config.get("model", {}).get("torch_dtype", "bfloat16") if config else "bfloat16"
    )
    torch_dtype = resolve_dtype(torch_dtype_str)

    quant_dict = config.get("quantization", {}) if config else {"load_in_4bit": True}
    quantization_config = build_quantization_config(quant_dict)

    trust_remote_code = (
        config.get("model", {}).get("trust_remote_code", True) if config else True
    )

    logger.info(f"Loading tokenizer for: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            padding_side="left",
        )
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "gated" in err_msg.lower() or "unauthorized" in err_msg.lower():
            raise PermissionError(
                f"Authentication error loading tokenizer for gated model '{model_name}'. "
                "Ensure Hugging Face token is set via HF_TOKEN environment variable."
            ) from e
        raise RuntimeError(f"Failed to load tokenizer for '{model_name}': {e}") from e

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    logger.info(
        f"Loading model: {model_name} (quantization={'4-bit' if quantization_config else 'None'}, dtype={torch_dtype_str})"
    )

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "token": hf_token,
        "device_map": device_map,
    }

    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["torch_dtype"] = torch_dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f"CUDA Out of Memory while loading '{model_name}'. Target RTX 4090 (24GB) exceeded: {e}"
        ) from e
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "gated" in err_msg.lower() or "unauthorized" in err_msg.lower():
            raise PermissionError(
                f"Authentication error loading model weights for gated model '{model_name}'. "
                "Meta LLaMA 3.1 requires repository access approval and HF_TOKEN configured."
            ) from e
        raise RuntimeError(f"Failed to load model '{model_name}': {e}. Refusing to fall back to CPU.") from e

    # Verify model is indeed on CUDA
    devices_in_use = {p.device.type for p in model.parameters()}
    if enforce_cuda and "cuda" not in devices_in_use:
        raise RuntimeError(
            f"Model '{model_name}' was loaded onto {devices_in_use} instead of CUDA. "
            "Silent fallback to non-CUDA device is forbidden."
        )

    # Optional gradient checkpointing
    gradient_checkpointing = (
        config.get("training", {}).get("gradient_checkpointing", False) if config else False
    )
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for memory efficiency.")

    # PEFT support if configured
    peft_dict = config.get("peft", {}) if config else {}
    if peft_dict.get("enabled", False):
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if quantization_config is not None:
                model = prepare_model_for_kbit_training(model)
            
            lora_config = LoraConfig(
                r=peft_dict.get("r", 16),
                lora_alpha=peft_dict.get("lora_alpha", 32),
                lora_dropout=peft_dict.get("lora_dropout", 0.05),
                bias=peft_dict.get("bias", "none"),
                task_type=peft_dict.get("task_type", "CAUSAL_LM"),
                target_modules=peft_dict.get("target_modules"),
            )
            model = get_peft_model(model, lora_config)
            logger.info("PEFT LoRA adapter successfully attached.")
        except Exception as peft_err:
            logger.warning(f"PEFT setup skipped or encountered notice: {peft_err}")

    # Calculate memory footprint
    vram_allocated_gb = 0.0
    if torch.cuda.is_available():
        vram_allocated_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 3)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_name": model_name,
        "devices": list(devices_in_use),
        "vram_allocated_gb": vram_allocated_gb,
    }


def run_smoke_inference(
    model_bundle: Dict[str, Any],
    prompt: str = "Explain prompt sensitivity in two sentences.",
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    """Execute a single forward-pass generation smoke test to confirm model readiness."""
    import torch

    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]

    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    # Move inputs to model device
    first_param = next(model.parameters())
    inputs = {k: v.to(first_param.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    vram_after = 0.0
    if torch.cuda.is_available():
        vram_after = round(torch.cuda.memory_allocated() / (1024 ** 3), 3)

    return {
        "prompt": prompt,
        "generated_text": output_text,
        "tokens_generated": len(output_ids[0]) - inputs["input_ids"].shape[1],
        "vram_allocated_gb": vram_after,
    }
