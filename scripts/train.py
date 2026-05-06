#!/usr/bin/env python3
"""Config-driven YOLO training wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from experiment_artifacts import copy_critical_reports, extract_run_dir
from schemas.train_config import DatasetConfig, TrainConfig
from utils.config_loader import load_config, validate_config
from utils.logger import get_logger
from utils.versioning import get_next_version, promote_run_artifacts
from utils.metadata import build_training_metadata, write_metadata

logger = get_logger(__name__)


def build_train_kwargs(train_cfg: TrainConfig, data_cfg_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'data': str(data_cfg_path),
        'imgsz': train_cfg.imgsz,
        'epochs': train_cfg.epochs,
        'batch': train_cfg.batch,
        'workers': train_cfg.workers,
        'device': train_cfg.device,
        'half': train_cfg.half,
        'cos_lr': train_cfg.cos_lr,
        'patience': train_cfg.patience,
        'project': train_cfg.project,
        'name': train_cfg.name,
        'cache': train_cfg.cache,
    }
    if args.resume:
        kwargs['resume'] = args.resume
    if args.freeze is not None:
        kwargs['freeze'] = args.freeze
    
    # Class imbalance handle 
    kwargs['cls'] = 1.0     
    kwargs['mixup'] = 0.1
    
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description='Train YOLO model with project conventions')
    parser.add_argument('--config', type=Path, default=Path('configs/train_local.yaml'))
    parser.add_argument('--data-config', type=Path, default=Path('configs/dataset_rgb.yaml'))
    parser.add_argument('--model', default=None)
    parser.add_argument('--mode', default='local')
    parser.add_argument('--version-bump', choices=['major', 'minor', 'patch'], default='patch')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--finetune', default=None)
    parser.add_argument('--freeze', type=int, default=None)
    parser.add_argument('--focal-loss', action='store_true')
    parser.add_argument('--cos-lr', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    train_cfg = validate_config(load_config(args.config), TrainConfig)
    dataset_cfg = validate_config(load_config(args.data_config), DatasetConfig)
    train_cfg_dict = train_cfg.model_dump()
    if args.model:
        train_cfg_dict['model'] = args.model
    if args.cos_lr:
        train_cfg_dict['cos_lr'] = True
    train_cfg = TrainConfig.model_validate(train_cfg_dict)

    kwargs = build_train_kwargs(train_cfg, args.data_config, args)
    model_source = args.finetune or train_cfg.model

    logger.info('train_config_resolved', config=str(args.config), data_config=str(args.data_config), mode=args.mode)
    logger.info('dataset_info', dataset_path=dataset_cfg.path, nc=dataset_cfg.nc)

    if args.dry_run:
        logger.info('dry_run_ok', model=model_source, kwargs=kwargs)
        return

    model = YOLO(model_source)
    result = model.train(**kwargs)
    run_dir = extract_run_dir(result)

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / 'models'
    version = get_next_version(models_dir, args.version_bump)
    artifact_dir = promote_run_artifacts(run_dir, version, models_dir)
    report_dir = copy_critical_reports(run_dir=run_dir, root=project_root, version=version)

    metadata = build_training_metadata(
        version=version,
        args=args,
        results=None,
        run_dir=run_dir,
        git_info={'commit': 'unknown', 'branch': 'unknown', 'dirty': False},
    )
    write_metadata(artifact_dir / 'metadata.json', metadata)

    # Otomatik TensorRT (.engine) Export
    best_pt_path = artifact_dir / 'best.pt'
    if best_pt_path.exists():
        logger.info('exporting_to_tensorrt', path=str(best_pt_path))
        try:
            export_model = YOLO(best_pt_path)
            # Eğer eğitim CPU'da yapılmış olsa bile GPU varsa export için cihaz 0'ı (GPU) kullanmayı dene
            export_device = train_cfg.device if str(train_cfg.device) != "cpu" else 0
            export_model.export(format='engine', device=export_device, half=True, imgsz=train_cfg.imgsz)
            logger.info('tensorrt_export_success', engine_path=str(best_pt_path.with_suffix('.engine')))
        except Exception as e:
            logger.warning('tensorrt_export_failed', error=str(e), note="GPU/CUDA kurulu olmayabilir.")

    logger.info('training_complete', run_dir=str(run_dir), version=version, model_dir=str(artifact_dir), report_dir=str(report_dir))


if __name__ == '__main__':
    main()
