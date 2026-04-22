#!/usr/bin/env python3
"""Offline augmentation for YOLO datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A
import cv2

try:
    from utils.config_loader import load_config
    from utils.logger import get_logger
except Exception:  # pragma: no cover
    from scripts.utils.config_loader import load_config
    from scripts.utils.logger import get_logger

logger = get_logger(__name__)


def build_pipeline(config_path: Path) -> A.Compose:
    cfg = load_config(config_path)
    transforms = []
    for item in cfg.get('transforms', []):
        if not isinstance(item, dict):
            continue
        name, params = next(iter(item.items()))
        transform_cls = getattr(A, name)
        transforms.append(transform_cls(**params))
    return A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def _read_yolo_labels(label_path: Path) -> tuple[list[list[float]], list[int]]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    if not label_path.exists():
        return boxes, labels
    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        labels.append(int(parts[0]))
        boxes.append([float(v) for v in parts[1:]])
    return boxes, labels


def _write_yolo_labels(label_path: Path, boxes: list[list[float]], labels: list[int]) -> None:
    with label_path.open('w', encoding='utf-8') as f:
        for cls, box in zip(labels, boxes):
            f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")


def augment_dataset(source: Path, output: Path, pipeline_path: Path, target_count: int) -> None:
    pipe = build_pipeline(pipeline_path)
    img_dir = source / 'images'
    lbl_dir = source / 'labels'
    out_img = output / 'images'
    out_lbl = output / 'labels'
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in img_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and '_aug_' not in p.name])
    for idx, img_path in enumerate(images[:target_count]):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes, labels = _read_yolo_labels(lbl_dir / f'{img_path.stem}.txt')
        augmented = pipe(image=img, bboxes=boxes, class_labels=labels)

        out_name = f'{img_path.stem}_aug_{idx}{img_path.suffix.lower()}'
        cv2.imwrite(str(out_img / out_name), augmented['image'])
        _write_yolo_labels(out_lbl / f'{Path(out_name).stem}.txt', list(augmented['bboxes']), list(augmented['class_labels']))

    logger.info('augmentation_complete', source=str(source), output=str(output), generated=min(len(images), target_count))


def main() -> None:
    parser = argparse.ArgumentParser(description='Offline YOLO augmentation')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pipeline', type=Path, default=Path('configs/augmentation_pipeline.yaml'))
    parser.add_argument('--target-class', type=int, default=-1)
    parser.add_argument('--target-count', type=int, default=500)
    args = parser.parse_args()
    augment_dataset(args.source, args.output, args.pipeline, args.target_count)


if __name__ == '__main__':
    main()
