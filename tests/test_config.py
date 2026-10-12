"""Unit tests for configuration loading and validation."""

from pathlib import Path
import pytest
import yaml

from utils import load_and_validate_config


def test_valid_config_loads_successfully():
    """Verify that default config.yaml passes all schema and value checks."""
    config_path = Path("config.yaml")
    assert config_path.is_file(), "config.yaml must exist at project root."
    config = load_and_validate_config(config_path)

    assert config["project"]["name"] == "prompt-sensitivity-hybrid"
    assert config["hardware"]["target_gpu"] == "NVIDIA GeForce RTX 4090"
    assert config["hardware"]["min_vram_gb"] >= 24.0
    assert config["hardware"]["enforce_cuda"] is True

    # Validate model parameters
    assert "Qwen/Qwen2.5-7B-Instruct" in config["model"]["supported_models"]
    assert "meta-llama/Llama-3.1-8B-Instruct" in config["model"]["supported_models"]
    assert config["model"]["fallback_allowed"] is False

    # Validate quantization
    assert config["quantization"]["load_in_4bit"] is True
    assert config["quantization"]["bnb_4bit_quant_type"] in ["nf4", "fp4"]

    # Validate dataset split
    d = config["dataset"]
    assert pytest.approx(d["train_ratio"] + d["val_ratio"] + d["test_ratio"], rel=1e-5) == 1.0
    assert d["data_path"] == "data/paraphrased_prompts_axis_tagged_qa.csv"


def test_missing_config_raises_file_not_found(tmp_path):
    """Ensure missing config file raises FileNotFoundError."""
    non_existent = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_and_validate_config(non_existent)


def test_invalid_split_ratios_rejected(tmp_path):
    """Ensure split ratios that do not sum to 1.0 are rejected."""
    bad_config = {
        "project": {"name": "test", "seed": 42},
        "hardware": {"target_gpu": "RTX 4090", "min_vram_gb": 24.0, "enforce_cuda": True},
        "model": {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "supported_models": ["Qwen/Qwen2.5-7B-Instruct"],
            "fallback_allowed": False,
        },
        "quantization": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True},
        "training": {"batch_size": 1, "gradient_accumulation_steps": 4, "gradient_checkpointing": True},
        "peft": {"enabled": False},
        "dataset": {"data_path": "dummy.jsonl", "cluster_key": "cluster_id", "train_ratio": 0.5, "val_ratio": 0.2, "test_ratio": 0.1},
    }
    cfg_file = tmp_path / "bad_split.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(bad_config, f)

    with pytest.raises(ValueError, match="split ratios must sum to 1.0"):
        load_and_validate_config(cfg_file)


def test_fallback_allowed_true_rejected(tmp_path):
    """Ensure that setting fallback_allowed=True is strictly prohibited and raises ValueError."""
    bad_config = {
        "project": {"name": "test", "seed": 42},
        "hardware": {"target_gpu": "RTX 4090", "min_vram_gb": 24.0, "enforce_cuda": True},
        "model": {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "supported_models": ["Qwen/Qwen2.5-7B-Instruct"],
            "fallback_allowed": True,  # Forbidden
        },
        "quantization": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True},
        "training": {"batch_size": 1, "gradient_accumulation_steps": 4, "gradient_checkpointing": True},
        "peft": {"enabled": False},
        "dataset": {"data_path": "dummy.jsonl", "cluster_key": "cluster_id", "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
    }
    cfg_file = tmp_path / "bad_fallback.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(bad_config, f)

    with pytest.raises(ValueError, match="fallback_allowed must be false"):
        load_and_validate_config(cfg_file)


def test_unsupported_model_rejected(tmp_path):
    """Ensure that configuring an unsupported model name fails validation."""
    bad_config = {
        "project": {"name": "test", "seed": 42},
        "hardware": {"target_gpu": "RTX 4090", "min_vram_gb": 24.0, "enforce_cuda": True},
        "model": {
            "name": "unsupported/random-model-7b",
            "supported_models": ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
            "fallback_allowed": False,
        },
        "quantization": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True},
        "training": {"batch_size": 1, "gradient_accumulation_steps": 4, "gradient_checkpointing": True},
        "peft": {"enabled": False},
        "dataset": {"data_path": "dummy.jsonl", "cluster_key": "cluster_id", "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
    }
    cfg_file = tmp_path / "bad_model.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(bad_config, f)

    with pytest.raises(ValueError, match="is not in supported_models"):
        load_and_validate_config(cfg_file)
