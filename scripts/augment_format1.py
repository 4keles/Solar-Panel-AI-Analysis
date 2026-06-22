#!/usr/bin/env python3
"""
Format1 augmentation script.
Format1 kaynaklı görüntüleri (hex-prefix isim pattern) unifiedV2_dataset/train içinden
tespit edip N kat augment ederek aynı train klasörüne ekler.

Usage:
  python scripts/augment_format1.py [--multiplier 5] [--seed 42]
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import albumentations as A
from utils.logger import get_logger

logger = get_logger(__name__)

BASE    = Path(__file__).resolve().parent.parent
DATASET = BASE / "data/processed_data/solar/unifiedV2_dataset"
TRAIN_IMG = DATASET / "train/images"
TRAIN_LBL = DATASET / "train/labels"

# Format1 dosyaları 8-char hex + tire ile başlar (örn: 006b381d-Physical_24.jpg)
F1_PATTERN = r"^[0-9a-f]{8}-"

TARGET_NAMES = ["bird_drop", "bird_feather", "physical_damage",
                "dust_partical", "leaf", "snow"]


def is_format1(stem: str) -> bool:
    import re
    return bool(re.match(r"^[0-9a-f]{8}-", stem))


def build_transform(seed: int | None = None) -> A.Compose:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.SafeRotate(limit=20, p=0.6, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=25, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.CLAHE(clip_limit=3.0, p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.25),
        A.RandomScale(scale_limit=(-0.2, 0.3), p=0.4),
        A.PadIfNeeded(
            min_height=640, min_width=640,
            border_mode=cv2.BORDER_CONSTANT, value=114,
            p=1.0,
        ),
        A.RandomCrop(height=640, width=640, p=1.0),
        A.CoarseDropout(
            num_holes_range=(2, 6),
            hole_height_range=(20, 60),
            hole_width_range=(20, 60),
            fill=114, p=0.3,
        ),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.35,
        clip=True,
    ))


def read_label(lbl_path: Path) -> tuple[list[int], list[list[float]]]:
    """Returns (class_ids, bboxes_yolo) for a label file."""
    class_ids, bboxes = [], []
    if not lbl_path.exists():
        return class_ids, bboxes
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_ids.append(int(float(parts[0])))
        bboxes.append([float(x) for x in parts[1:5]])
    return class_ids, bboxes


def write_label(lbl_path: Path, class_ids: list[int], bboxes: list[list[float]]) -> None:
    lines = [f"{c} {' '.join(f'{v:.6f}' for v in bb)}"
             for c, bb in zip(class_ids, bboxes)]
    lbl_path.write_text("\n".join(lines))


def augment_one(img_path: Path, lbl_path: Path, out_img_dir: Path,
                out_lbl_dir: Path, aug_idx: int, transform: A.Compose) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    class_ids, bboxes = read_label(lbl_path)

    try:
        result = transform(image=img_rgb, bboxes=bboxes, class_labels=class_ids)
    except Exception as e:
        logger.warning("aug_error", file=img_path.name, error=str(e))
        return False

    aug_img   = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
    aug_bboxes = result["bboxes"]
    aug_cls    = result["class_labels"]

    stem = img_path.stem
    ext  = img_path.suffix
    out_img = out_img_dir / f"{stem}_f1aug{aug_idx}{ext}"
    out_lbl = out_lbl_dir / f"{stem}_f1aug{aug_idx}.txt"

    cv2.imwrite(str(out_img), aug_img)
    write_label(out_lbl, list(aug_cls), [list(b) for b in aug_bboxes])
    return True


def main(multiplier: int = 5, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # Format1 kaynaklı train görüntülerini bul
    f1_imgs = sorted([p for p in TRAIN_IMG.glob("*")
                      if is_format1(p.stem) and p.suffix.lower()
                      in {".jpg", ".jpeg", ".png"}])

    logger.info("format1_found", count=len(f1_imgs))
    if not f1_imgs:
        logger.warning("no_format1_files_found")
        return

    # Her sınıf için mevcut sayıyı logla
    cls_count: dict[int, int] = {}
    for img in f1_imgs:
        lbl = TRAIN_LBL / (img.stem + ".txt")
        for cid, _ in zip(*read_label(lbl)):
            cls_count[cid] = cls_count.get(cid, 0) + 1
    logger.info("class_dist_before_aug",
                dist={TARGET_NAMES[k]: v for k, v in sorted(cls_count.items())
                      if k < len(TARGET_NAMES)})

    transform = build_transform(seed=seed)
    success = 0
    fail    = 0

    for i, img_path in enumerate(f1_imgs):
        lbl_path = TRAIN_LBL / (img_path.stem + ".txt")
        for aug_idx in range(1, multiplier + 1):
            ok = augment_one(img_path, lbl_path, TRAIN_IMG, TRAIN_LBL,
                             aug_idx, transform)
            if ok:
                success += 1
            else:
                fail += 1
        if (i + 1) % 100 == 0:
            logger.info("progress", done=i + 1, total=len(f1_imgs),
                        success=success, fail=fail)

    logger.info("augmentation_complete",
                f1_source=len(f1_imgs), multiplier=multiplier,
                new_images=success, failed=fail,
                total_train_approx=len(f1_imgs) * (1 + multiplier) +
                                   len(list(TRAIN_IMG.glob("*"))) - len(f1_imgs))

    # Cache'i sil — YOLO yeniden tarar
    cache = DATASET / "train/labels.cache"
    if cache.exists():
        cache.unlink()
        logger.info("cache_cleared")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiplier", type=int, default=5,
                    help="Her format1 görüntüsü için kaç aug kopyası")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(multiplier=args.multiplier, seed=args.seed)
