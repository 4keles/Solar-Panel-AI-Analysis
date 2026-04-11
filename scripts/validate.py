#!/usr/bin/env python3
"""Validation wrapper producing contract-compatible summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from experiment_artifacts import copy_critical_reports, extract_run_dir
from utils.logger import get_logger

logger = get_logger(__name__)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def build_summary(version: str, split: str, conf: float, iou: float, results: Any) -> dict[str, Any]:
    metrics = getattr(results, 'results_dict', {}) if hasattr(results, 'results_dict') else {}
    summary = {
        'model_version': version,
        'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'split': split,
        'thresholds': {'conf': conf, 'iou': iou},
        'overall': {
            'mAP50': _safe_float(metrics.get('metrics/mAP50(B)')),
            'mAP50_95': _safe_float(metrics.get('metrics/mAP50-95(B)')),
            'precision': _safe_float(metrics.get('metrics/precision(B)')),
            'recall': _safe_float(metrics.get('metrics/recall(B)')),
            'f1': 0.0,
        },
        'per_class': {},
    }
    p = summary['overall']['precision']
    r = summary['overall']['recall']
    summary['overall']['f1'] = (2 * p * r / (p + r)) if (p + r) else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate trained YOLO model')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('configs/dataset_rgb.yaml'))
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--version', default='v0.0.0')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    results = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch, device=args.device)
    run_dir = extract_run_dir(results)

    summary = build_summary(args.version, args.split, args.conf, args.iou, results)
    root = Path(__file__).resolve().parent.parent
    report_dir = copy_critical_reports(run_dir=run_dir, root=root, version=args.version)

    summary_path = report_dir / 'val_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('validation_complete', summary_path=str(summary_path), report_dir=str(report_dir))


if __name__ == '__main__':
    main()
