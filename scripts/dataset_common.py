"""Shared paths and registry loading for dataset prep and YOLO data.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def registry_path() -> Path:
    return Path(__file__).resolve().parent / "dataset_registry.yaml"


def load_registry() -> dict[str, Any]:
    with open(registry_path(), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("dataset_registry.yaml must be a mapping at the top level")
    return data


def get_dataset_entry(registry: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    datasets = registry.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("registry missing 'datasets' mapping")
    entry = datasets.get(dataset_id)
    if not isinstance(entry, dict):
        raise KeyError(f"Unknown dataset id: {dataset_id!r}")
    return entry


def resolve_under(root: Path, rel: str) -> Path:
    return (root / rel).resolve()
