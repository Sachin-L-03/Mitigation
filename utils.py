"""Utility functions for prompt-sensitivity-hybrid.

Provides reproducible seeding, structured logging, configuration validation,
system environment inspection, and experiment artifact management.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def set_seed(seed: int = 42) -> int:
    """Set seeds across Python, NumPy, and PyTorch for deterministic reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def setup_logger(
    name: str = "prompt_sensitivity",
    log_dir: str | Path = "outputs/logs",
    log_filename: str = "run.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure unified console and file logging."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    full_log_file = log_path / log_filename

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on re-initialization
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(full_log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_and_validate_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and validate the YAML configuration schema.

    Strictly validates required keys, types, value constraints, and model specifications.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {path}: {e}") from e

    if not isinstance(config, dict):
        raise ValueError("Configuration file root must be a YAML dictionary mapping.")

    required_sections = ["project", "hardware", "model", "quantization", "training", "peft", "dataset"]
    for sec in required_sections:
        if sec not in config or not isinstance(config[sec], dict):
            raise ValueError(f"Missing or invalid required configuration section: '{sec}'")

    # Project validation
    p = config["project"]
    if not isinstance(p.get("name"), str) or not p["name"].strip():
        raise ValueError("project.name must be a non-empty string.")
    if not isinstance(p.get("seed"), int):
        raise ValueError("project.seed must be an integer.")

    # Hardware validation
    hw = config["hardware"]
    if not isinstance(hw.get("target_gpu"), str):
        raise ValueError("hardware.target_gpu must be a string describing the target GPU.")
    if not isinstance(hw.get("min_vram_gb"), (int, float)) or hw["min_vram_gb"] <= 0:
        raise ValueError("hardware.min_vram_gb must be a positive number.")
    if not isinstance(hw.get("enforce_cuda"), bool):
        raise ValueError("hardware.enforce_cuda must be a boolean.")

    # Model validation
    m = config["model"]
    if not isinstance(m.get("name"), str) or not m["name"].strip():
        raise ValueError("model.name must be a non-empty string.")
    supported_models = m.get("supported_models", [])
    if not isinstance(supported_models, list) or not supported_models:
        raise ValueError("model.supported_models must be a non-empty list.")
    if m["name"] not in supported_models:
        raise ValueError(
            f"Configured model '{m['name']}' is not in supported_models: {supported_models}"
        )
    if not isinstance(m.get("fallback_allowed"), bool):
        raise ValueError("model.fallback_allowed must be a boolean.")
    if m["fallback_allowed"]:
        raise ValueError("model.fallback_allowed must be false (silent fallbacks are strictly prohibited).")

    # Quantization validation
    q = config["quantization"]
    if not isinstance(q.get("load_in_4bit"), bool):
        raise ValueError("quantization.load_in_4bit must be a boolean.")
    if q.get("bnb_4bit_quant_type") not in ["nf4", "fp4"]:
        raise ValueError("quantization.bnb_4bit_quant_type must be either 'nf4' or 'fp4'.")
    if not isinstance(q.get("bnb_4bit_use_double_quant"), bool):
        raise ValueError("quantization.bnb_4bit_use_double_quant must be a boolean.")

    # Training validation
    t = config["training"]
    if not isinstance(t.get("batch_size"), int) or t["batch_size"] <= 0:
        raise ValueError("training.batch_size must be a positive integer.")
    if not isinstance(t.get("gradient_accumulation_steps"), int) or t["gradient_accumulation_steps"] <= 0:
        raise ValueError("training.gradient_accumulation_steps must be a positive integer.")
    if not isinstance(t.get("gradient_checkpointing"), bool):
        raise ValueError("training.gradient_checkpointing must be a boolean.")

    # Dataset validation
    d = config["dataset"]
    train_r = d.get("train_ratio", 0.0)
    val_r = d.get("val_ratio", 0.0)
    test_r = d.get("test_ratio", 0.0)
    total_r = train_r + val_r + test_r
    if abs(total_r - 1.0) > 1e-5:
        raise ValueError(
            f"Dataset split ratios must sum to 1.0 (train={train_r}, val={val_r}, test={test_r}, sum={total_r})"
        )
    if train_r <= 0 or val_r < 0 or test_r < 0:
        raise ValueError("Dataset split ratios must be non-negative with positive train_ratio.")
    if not isinstance(d.get("cluster_key"), str) or not d["cluster_key"].strip():
        raise ValueError("dataset.cluster_key must be a non-empty string.")

    return config


def inspect_system_environment(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Inspect environment including OS, Python, PyTorch, CUDA, GPU model, VRAM, and packages."""
    env_info: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "target_check_passed": False,
        "quantization_support": False,
        "packages": {},
        "issues": [],
    }

    # Inspect installed packages
    for pkg_name in ["torch", "transformers", "accelerate", "bitsandbytes", "peft", "yaml"]:
        try:
            mod = __import__(pkg_name)
            env_info["packages"][pkg_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            env_info["packages"][pkg_name] = "not installed"
            env_info["issues"].append(f"Package '{pkg_name}' is not installed.")

    # PyTorch and CUDA checks
    try:
        import torch
        env_info["cuda_available"] = torch.cuda.is_available()
        env_info["torch_version"] = torch.__version__
        env_info["torch_cuda_version"] = torch.version.cuda

        if torch.cuda.is_available():
            dev_count = torch.cuda.device_count()
            env_info["device_count"] = dev_count
            for i in range(dev_count):
                props = torch.cuda.get_device_properties(i)
                total_vram_gb = round(props.total_memory / (1024 ** 3), 2)
                device_info = {
                    "index": i,
                    "name": props.name,
                    "total_vram_gb": total_vram_gb,
                    "major_capability": props.major,
                    "minor_capability": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
                env_info["devices"].append(device_info)

            # Target RTX 4090 evaluation
            primary_gpu = env_info["devices"][0]
            target_gpu_name = config["hardware"]["target_gpu"] if config else "RTX 4090"
            min_vram = config["hardware"]["min_vram_gb"] if config else 24.0

            gpu_matches = target_gpu_name.lower() in primary_gpu["name"].lower()
            vram_sufficient = primary_gpu["total_vram_gb"] >= (min_vram - 1.0)  # Allowance for sys reserve

            env_info["target_check_passed"] = bool(gpu_matches and vram_sufficient)
            if not gpu_matches:
                env_info["issues"].append(
                    f"GPU name '{primary_gpu['name']}' does not match target '{target_gpu_name}'."
                )
            if not vram_sufficient:
                env_info["issues"].append(
                    f"GPU VRAM {primary_gpu['total_vram_gb']} GB is below target {min_vram} GB."
                )
        else:
            env_info["issues"].append("CUDA is not available in PyTorch.")
    except Exception as e:
        env_info["issues"].append(f"Error inspecting PyTorch/CUDA: {e}")

    # Quantization / BitsAndBytes checks
    try:
        import bitsandbytes as bnb
        env_info["quantization_support"] = True
        env_info["bitsandbytes_version"] = getattr(bnb, "__version__", "unknown")
    except Exception as e:
        env_info["quantization_support"] = False
        env_info["issues"].append(f"BitsAndBytes quantization check failed: {e}")

    return env_info


def save_experiment_manifest(
    output_dir: str | Path,
    config: Dict[str, Any],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save an execution manifest recording config, environment, and metadata for reproducibility."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_file = out_path / "experiment_manifest.json"

    manifest_data = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": config,
        "metadata": extra_metadata or {},
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    return manifest_file
