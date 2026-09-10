"""Configuration loading utilities for ClaimGuard.

Deliberately dependency-light: only `pyyaml` and the standard library. This
module must be importable and usable without torch/transformers/GPU access,
so config-only smoke tests can run without touching models or the GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# src/claimguard/config.py -> src/claimguard -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

REQUIRED_DEFAULT_KEYS = {
    "seed",
    "device",
    "logging",
    "paths",
    "retrieval",
    "reranking",
    "correction",
    "context",
    "batch_sizes",
    "precision",
}


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing, empty, or malformed."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file did not parse to a mapping: {path}")
    return data


def load_default_config(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load configs/default.yaml."""
    return _load_yaml(config_dir / "default.yaml")


def load_models_config(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load configs/models.yaml."""
    return _load_yaml(config_dir / "models.yaml")


def load_experiments_config(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load configs/experiments.yaml."""
    return _load_yaml(config_dir / "experiments.yaml")


def load_verifier_baseline_config(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load configs/verifier_baseline.yaml (Step 9's preliminary,
    HaluEval-only baseline config)."""
    return _load_yaml(config_dir / "verifier_baseline.yaml")


def load_verifier_binary_final_config(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load configs/verifier_binary_final.yaml (Step 12's final binary
    FEVER+HaluEval verifier config)."""
    return _load_yaml(config_dir / "verifier_binary_final.yaml")


def load_reranker_baseline_config(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load configs/reranker_baseline.yaml (Step 14's BGE reranker +
    two-stage retrieval pipeline config)."""
    return _load_yaml(config_dir / "reranker_baseline.yaml")


def load_integration_baseline_config(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load configs/integration_baseline.yaml (Step 15's retrieval +
    reranking + binary verifier integration config)."""
    return _load_yaml(config_dir / "integration_baseline.yaml")


def load_decision_policy_config(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load configs/decision_policy.yaml (Step 16's verifier calibration +
    deterministic evidence-decision-policy config)."""
    return _load_yaml(config_dir / "decision_policy.yaml")


def load_generator_baseline_config(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load configs/generator_baseline.yaml (Step 17's Qwen3-8B generator
    config)."""
    return _load_yaml(config_dir / "generator_baseline.yaml")


def load_correction_baseline_config(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load configs/correction_baseline.yaml (Step 18's bounded
    correction/regeneration loop config)."""
    return _load_yaml(config_dir / "correction_baseline.yaml")


def validate_default_config(config: dict[str, Any]) -> None:
    """Raise ConfigError if any required top-level key is missing."""
    missing = REQUIRED_DEFAULT_KEYS - config.keys()
    if missing:
        raise ConfigError(f"Missing required config keys: {sorted(missing)}")


def resolve_path(relative_path: str, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a project-relative path against the project root."""
    return (root / relative_path).resolve()
