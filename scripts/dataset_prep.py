#!/usr/bin/env python3
"""Copy or symlink raw YOLO splits into processed_data with validation."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from dataset_common import get_dataset_entry, load_registry, project_root, resolve_under
from yolo_data_yaml import write_for_registry_dataset

try:
    from utils.logger import get_logger
except Exception:  # pragma: no cover
    from scripts.utils.logger import get_logger

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
logger = get_logger(__name__)


def _iter_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXT]


def _validate_label_lines(label_path: Path, nc: int) -> list[str]:
    errors: list[str] = []
    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{label_path}: read error: {exc}"]
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cid = int(parts[0])
        except ValueError:
            errors.append(f"{label_path}:{lineno}: non-integer class {parts[0]!r}")
            continue
        if not (0 <= cid < nc):
            errors.append(f"{label_path}:{lineno}: class {cid} not in [0, {nc - 1}]")
    return errors


def _scan_label_classes(labels_dir: Path) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    if not labels_dir.is_dir():
        return counts
    for txt in sorted(labels_dir.glob("*.txt")):
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                counts[int(parts[0])] += 1
            except (ValueError, IndexError):
                continue
    return dict(counts)


def prepare_dataset(
    dataset_id: str,
    *,
    use_symlinks: bool = False,
    scan_labels: bool = False,
    strict_labels: bool = False,
    write_yaml: bool = True,
) -> Path:
    root = project_root()
    registry = load_registry()
    entry = get_dataset_entry(registry, dataset_id)

    raw_root = resolve_under(root, entry["raw_root"])
    processed_root = resolve_under(root, registry["processed_data_root"])
    output_subdir = entry["output_subdir"]
    out_dataset = (processed_root / output_subdir).resolve()

    class_names: list[str] = list(entry["class_names"])
    nc = len(class_names)
    split_map: dict[str, str] = dict(entry["split_map"])
    images_sub = entry.get("images_subdir", "images")
    labels_sub = entry.get("labels_subdir", "labels")

    if out_dataset.exists() and not use_symlinks:
        shutil.rmtree(out_dataset)
    out_dataset.mkdir(parents=True, exist_ok=True)

    target_names = {"train": "train", "val": "val", "test": "test"}

    for logical in ("train", "val"):
        if logical not in split_map:
            raise KeyError(f"split_map must include {logical!r}")

    for logical, source_folder in split_map.items():
        src_split = (raw_root / source_folder).resolve()
        dst_split = (out_dataset / target_names.get(logical, logical)).resolve()

        if not src_split.is_dir():
            if logical in {"train", "val"}:
                raise FileNotFoundError(f"Required split missing: {src_split}")
            logger.info("optional_split_missing", split=logical, source=str(src_split))
            continue

        if use_symlinks:
            if dst_split.exists() or dst_split.is_symlink():
                dst_split.unlink()
            dst_split.symlink_to(src_split, target_is_directory=True)
        else:
            shutil.copytree(src_split, dst_split, dirs_exist_ok=True)

        check_images = dst_split / images_sub
        check_labels = dst_split / labels_sub

        if check_images.is_dir() and check_labels.is_dir():
            images = _iter_images(check_images)
            label_stems = {p.stem for p in check_labels.glob("*.txt")}
            image_stems = {p.stem for p in images}
            for stem in sorted(image_stems - label_stems):
                logger.info("image_without_label", split=logical, stem=stem)
            for stem in sorted(label_stems - image_stems):
                logger.info("label_without_image", split=logical, stem=stem)

            all_errors: list[str] = []
            for lbl in sorted(check_labels.glob("*.txt")):
                all_errors.extend(_validate_label_lines(lbl, nc))
            if all_errors:
                for msg in all_errors[:50]:
                    logger.warning("label_issue", message=msg)
                if strict_labels:
                    raise ValueError("Strict label validation failed")

        if scan_labels:
            counts = _scan_label_classes(check_labels)
            logger.info("class_counts", split=logical, counts=counts)

    if write_yaml:
        yaml_path = write_for_registry_dataset(dataset_id, registry=registry, check_only=False)
        logger.info("dataset_yaml_written", path=str(yaml_path))

    return out_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare processed YOLO dataset from raw data")
    parser.add_argument("--dataset", required=True, help="Registry dataset id (e.g. mvp_test)")
    parser.add_argument("--link", action="store_true", help="Symlink split folders instead of copy")
    parser.add_argument("--scan-labels", action="store_true", help="Print class histogram")
    parser.add_argument("--strict-labels", action="store_true", help="Fail on invalid class id")
    parser.add_argument("--no-yaml", action="store_true", help="Skip writing data yaml")
    args = parser.parse_args()

    prepare_dataset(
        args.dataset,
        use_symlinks=args.link,
        scan_labels=args.scan_labels,
        strict_labels=args.strict_labels,
        write_yaml=not args.no_yaml,
    )


if __name__ == "__main__":
    main()
