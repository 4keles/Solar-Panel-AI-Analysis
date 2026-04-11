"""Shared paths and registry loading for dataset prep and YOLO data.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from utils.config_loader import load_config
except Exception:  # pragma: no cover
    from scripts.utils.config_loader import load_config


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def registry_path() -> Path:
    return Path(__file__).resolve().parent / "dataset_registry.yaml"


def load_registry() -> dict[str, Any]:
    data = load_config(registry_path())
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
