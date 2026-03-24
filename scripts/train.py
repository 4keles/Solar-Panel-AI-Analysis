#!/usr/bin/env python3
"""Train Ultralytics YOLO11 with registry/versioning/report export."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from experiment_artifacts import (
    copy_best_weights_to_registry,
    copy_critical_reports,
    extract_metrics,
    extract_run_dir,
    project_root,
    resolve_path,
    save_model_metadata,
)


def maybe_enable_wandb(
    *,
    enable_wandb: bool,
    wandb_project: str | None,
    wandb_run_name: str | None,
) -> bool:
    """Enable W&B tracking if requested and installed."""
    if not enable_wandb:
        return False
    try:
        import wandb  # type: ignore
    except Exception:
        print("Warning: --enable-wandb set but wandb is not installed. Run: uv add wandb")
        return False

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name:
        os.environ["WANDB_NAME"] = wandb_run_name
    os.environ.setdefault("WANDB_MODE", "online")
    try:
        # Safe no-op if already authenticated.
        wandb.ensure_configured()
    except Exception:
        pass
    return True


def resolve_finetune_weights(root: Path, finetune_from: str | None) -> Path | None:
    if not finetune_from:
        return None
    as_path = Path(finetune_from)
    if as_path.suffix == ".pt" or as_path.exists():
        resolved = resolve_path(as_path, root)
    else:
        resolved = (root / "models" / finetune_from / "best.pt").resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Finetune checkpoint not found: {resolved}")
    return resolved


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO11 detection training")
    ap.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Ultralytics data.yaml (e.g. data/processed_data/mvp_test_v1.data.yaml)",
    )
    ap.add_argument("--model", default="yolo11n.pt", help="Checkpoint or model name")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 = AutoBatch)",
    )
    ap.add_argument("--device", default=None, help="cuda device, e.g. 0 or cpu")
    ap.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Ultralytics project directory (default: runs/detect under cwd)",
    )
    ap.add_argument("--name", default="solar_mvp", help="Run name under project")
    ap.add_argument(
        "--version",
        default="v1.0.0",
        help="Model and report version folder, e.g. v1.0.0",
    )
    ap.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume previous training (optional path to last.pt or run directory)",
    )
    ap.add_argument(
        "--finetune",
        nargs="?",
        const="v1.0.0",
        default=None,
        help="Finetune from a model version (e.g. v1.0.0) or explicit .pt path",
    )
    ap.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Freeze first N layers (used with --finetune)",
    )
    ap.add_argument(
        "--focal-loss",
        action="store_true",
        help="Enable focal-loss style training for class imbalance",
    )
    ap.add_argument(
        "--cos-lr",
        action="store_true",
        help="Enable cosine learning rate schedule",
    )
    ap.add_argument("--enable-wandb", action="store_true", help="Enable W&B logging if installed")
    ap.add_argument("--wandb-project", default="solar_panel_od", help="W&B project name")
    ap.add_argument("--wandb-run-name", default=None, help="Optional W&B run name")
    args = ap.parse_args()

    root = project_root()
    data_yaml = resolve_path(args.data, root)
    if args.resume and args.finetune:
        raise ValueError("--resume and --finetune cannot be used together")

    maybe_enable_wandb(
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    resume_arg: str | bool | None = None
    model_source = args.model
    if args.resume:
        if args.resume == "auto":
            resume_arg = True
        else:
            resume_target = resolve_path(Path(args.resume), root)
            if resume_target.is_dir():
                resume_target = resume_target / "weights" / "last.pt"
            resume_arg = str(resume_target)
            model_source = str(resume_target)

    finetune_weights = resolve_finetune_weights(root, args.finetune)
    if finetune_weights is not None:
        model_source = str(finetune_weights)

    train_kw: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "name": args.name,
    }
    if args.device is not None:
        train_kw["device"] = args.device
    if args.project is not None:
        train_kw["project"] = str(resolve_path(args.project, root))
    if resume_arg is not None:
        train_kw["resume"] = resume_arg
    if finetune_weights is not None:
        train_kw["freeze"] = args.freeze
    if args.focal_loss:
        train_kw["fl_gamma"] = 1.5
    if args.cos_lr:
        train_kw["cos_lr"] = True

    model = YOLO(model_source)
    results = model.train(**train_kw)

    run_dir = extract_run_dir(results)
    metrics = extract_metrics(results)
    registry_best, version_dir = copy_best_weights_to_registry(run_dir=run_dir, root=root, version=args.version)
    copy_critical_reports(run_dir=run_dir, root=root, version=args.version)
    meta_path = save_model_metadata(
        model_version_dir=version_dir,
        metrics=metrics,
        source_run_dir=run_dir,
        data_yaml=data_yaml,
        weights_source=registry_best,
    )
    print(f"Training completed. Versioned best weights: {registry_best}")
    print(f"Metadata written: {meta_path}")
    print(f"Reports exported to: {(root / 'reports' / args.version).resolve()}")


if __name__ == "__main__":
    main()
