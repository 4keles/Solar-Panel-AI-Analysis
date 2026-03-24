#!/usr/bin/env python3
"""Run Ultralytics validation and export enterprise reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from experiment_artifacts import (
    copy_critical_reports,
    extract_metrics,
    extract_run_dir,
    project_root,
    resolve_path,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO validation / metrics")
    ap.add_argument("--weights", type=Path, required=True, help="Trained .pt weights")
    ap.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Ultralytics data.yaml used for training",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", default="val", choices=("train", "val", "test"), help="Which split to evaluate")
    ap.add_argument(
        "--version",
        default="v1.0.0",
        help="Report version folder, e.g. v1.0.0",
    )
    args = ap.parse_args()

    root = project_root()
    weights = resolve_path(args.weights, root)
    data_yaml = resolve_path(args.data, root)

    val_kw: dict[str, Any] = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "split": args.split,
    }
    if args.device is not None:
        val_kw["device"] = args.device

    model = YOLO(str(weights))
    results = model.val(**val_kw)
    run_dir = extract_run_dir(results)
    metrics = extract_metrics(results)
    report_dir = copy_critical_reports(run_dir=run_dir, root=root, version=args.version)
    print(f"Validation metrics: {metrics}")
    print(f"Reports exported to: {report_dir}")


if __name__ == "__main__":
    main()
