"""Unit tests for model loading interface and constraints."""

from unittest.mock import patch
import pytest
import torch

from model import (
    SUPPORTED_MODELS,
    build_quantization_config,
    check_cuda_prerequisites,
    resolve_dtype,
    validate_model_target,
)


def test_supported_models():
    """Verify that both Qwen2.5 and LLaMA-3.1 are in SUPPORTED_MODELS."""
    assert "Qwen/Qwen2.5-7B-Instruct" in SUPPORTED_MODELS
    assert "meta-llama/Llama-3.1-8B-Instruct" in SUPPORTED_MODELS


def test_validate_model_target():
    """Verify target validation succeeds on supported models and fails on unlisted ones."""
    validate_model_target("Qwen/Qwen2.5-7B-Instruct")
    validate_model_target("meta-llama/Llama-3.1-8B-Instruct")

    with pytest.raises(ValueError, match="is not supported"):
        validate_model_target("unsupported-vendor/unknown-model")


def test_resolve_dtype():
    """Verify resolution of precision dtypes."""
    assert resolve_dtype("bfloat16") == torch.bfloat16
    assert resolve_dtype("float16") == torch.float16
    assert resolve_dtype("float32") == torch.float32

    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_dtype("int8")


def test_build_quantization_config():
    """Verify building 4-bit BitsAndBytesConfig."""
    quant_dict = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }
    bnb_cfg = build_quantization_config(quant_dict)
    assert bnb_cfg is not None
    assert bnb_cfg.load_in_4bit is True
    assert bnb_cfg.bnb_4bit_quant_type == "nf4"
    assert bnb_cfg.bnb_4bit_use_double_quant is True
    assert bnb_cfg.bnb_4bit_compute_dtype == torch.bfloat16

    # Test disabled quantization returns None
    assert build_quantization_config({"load_in_4bit": False}) is None


def test_cuda_prerequisites_enforcement():
    """Verify that if CUDA is not available, check_cuda_prerequisites strictly raises RuntimeError."""
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="Silent CPU fallback is prohibited"):
            check_cuda_prerequisites(enforce_cuda=True)
