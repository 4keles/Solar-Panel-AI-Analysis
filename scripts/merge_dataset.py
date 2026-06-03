#!/usr/bin/env python3
"""Merge captured and verified images/labels into a finetuning dataset structure."""

import argparse
import random
import shutil
import time
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


def merge_dataset(source_dir: Path, target_base: Path, split_ratio: float = 0.8) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        logger.error("source_directory_not_found", path=str(source_dir))
        return

    # Find all images that have a corresponding .txt file
    images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpeg"))
    valid_pairs = []
    
    for img_path in images:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            valid_pairs.append((img_path, txt_path))
            
    if not valid_pairs:
        logger.warning("no_verified_data_found", path=str(source_dir))
        return

    logger.info("found_valid_pairs", count=len(valid_pairs))
    
    # Shuffle for random split
    random.seed(42)
    random.shuffle(valid_pairs)
    
    train_count = int(len(valid_pairs) * split_ratio)
    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:]
    
    # Setup target directories
    img_train_dir = target_base / "images" / "train"
    img_val_dir = target_base / "images" / "val"
    lbl_train_dir = target_base / "labels" / "train"
    lbl_val_dir = target_base / "labels" / "val"
    
    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    archive_dir = source_dir / "archive" / time.strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    def process_split(pairs, img_dest, lbl_dest, split_name):
        for img_src, txt_src in pairs:
            # Copy to target
            shutil.copy(img_src, img_dest / img_src.name)
            shutil.copy(txt_src, lbl_dest / txt_src.name)
            
            # Move to archive to keep source clean
            shutil.move(str(img_src), str(archive_dir / img_src.name))
            shutil.move(str(txt_src), str(archive_dir / txt_src.name))
            
        logger.info("processed_split", split=split_name, count=len(pairs))

    process_split(train_pairs, img_train_dir, lbl_train_dir, "train")
    process_split(val_pairs, img_val_dir, lbl_val_dir, "val")
    
    logger.info("merge_complete", target=str(target_base), archived=str(archive_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge verified captures into dataset")
    parser.add_argument("--source", type=Path, default=Path("data/raw_data/captured"), help="Directory with captured images and .txt labels")
    parser.add_argument("--target", type=Path, default=Path("data/processed/finetune"), help="Target base dataset directory (e.g., data/processed/finetune)")
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio (0.0 to 1.0)")
    args = parser.parse_args()

    merge_dataset(args.source, args.target, args.split)


if __name__ == "__main__":
    main()
