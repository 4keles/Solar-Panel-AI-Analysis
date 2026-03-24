#!/usr/bin/env python3
"""Helpers for enterprise-style model registry and report export."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CRITICAL_REPORT_PATTERNS = (
    "confusion_matrix*.png",
    "F1_curve*.png",
    "PR_curve*.png",
    "P_curve*.png",
    "R_curve*.png",
    "results*.png",
    "val_batch*.jpg",
    "train_batch*.jpg",
    "predictions*.jpg",
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(p: Path, root: Path) -> Path:
    return p if p.is_absolute() else (root / p).resolve()


def to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metrics(result_obj: Any) -> dict[str, float | None]:
    """Extract core metrics from Ultralytics result-like objects."""
    metrics_src = getattr(result_obj, "results_dict", None)
    if not isinstance(metrics_src, dict):
        metrics_src = {}
    return {
        "precision": to_float_or_none(metrics_src.get("metrics/precision(B)")),
        "recall": to_float_or_none(metrics_src.get("metrics/recall(B)")),
        "mAP50": to_float_or_none(metrics_src.get("metrics/mAP50(B)")),
        "mAP50_95": to_float_or_none(metrics_src.get("metrics/mAP50-95(B)")),
    }


def ensure_model_version_dir(root: Path, version: str) -> Path:
    model_dir = (root / "models" / version).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_model_metadata(
    *,
    model_version_dir: Path,
    metrics: dict[str, float | None],
    source_run_dir: Path,
    data_yaml: Path,
    weights_source: Path,
) -> Path:
    meta_path = model_version_dir / "metadata.json"
    payload = {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source_run_dir": str(source_run_dir),
        "data_yaml": str(data_yaml),
        "weights_source": str(weights_source),
        "metrics": metrics,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def copy_best_weights_to_registry(
    *,
    run_dir: Path,
    root: Path,
    version: str,
) -> tuple[Path, Path]:
    src_best = (run_dir / "weights" / "best.pt").resolve()
    if not src_best.is_file():
        raise FileNotFoundError(f"Cannot find best weights at {src_best}")
    version_dir = ensure_model_version_dir(root, version)
    dst_best = version_dir / "best.pt"
    shutil.copy2(src_best, dst_best)
    return dst_best, version_dir


def copy_critical_reports(*, run_dir: Path, root: Path, version: str) -> Path:
    report_dir = (root / "reports" / version).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    copied: set[Path] = set()
    for pattern in CRITICAL_REPORT_PATTERNS:
        for src in run_dir.glob(pattern):
            if src.is_file():
                dst = report_dir / src.name
                shutil.copy2(src, dst)
                copied.add(dst)
    index_path = report_dir / "report_index.json"
    index_payload = {
        "source_run_dir": str(run_dir),
        "copied_files": sorted([p.name for p in copied]),
    }
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    return report_dir


def extract_run_dir(result_obj: Any) -> Path:
    run_dir = getattr(result_obj, "save_dir", None)
    if run_dir is None:
        raise ValueError("Could not determine run save_dir from Ultralytics results.")
    return Path(str(run_dir)).resolve()
