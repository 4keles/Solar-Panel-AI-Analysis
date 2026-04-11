"""Shared configuration loading and validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML/JSON config file into a dictionary."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(encoding='utf-8')
    if suffix in {'.yaml', '.yml'}:
        data = yaml.safe_load(text)
    elif suffix == '.json':
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Config must be mapping/object: {path}")
    return data


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, override has higher precedence."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_config(config: dict[str, Any], schema_class: type[BaseModel]) -> BaseModel:
    """Validate config dictionary against a Pydantic schema class."""
    return schema_class.model_validate(config)



def load_train_and_dataset_config(
    train_cfg_path: Path,
    dataset_cfg_path: Path,
    train_schema: type[BaseModel],
    dataset_schema: type[BaseModel],
) -> tuple[BaseModel, BaseModel]:
    """Load and validate train and dataset configs together."""
    train_cfg = validate_config(load_config(train_cfg_path), train_schema)
    dataset_cfg = validate_config(load_config(dataset_cfg_path), dataset_schema)
    return train_cfg, dataset_cfg
