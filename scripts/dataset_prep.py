#!/usr/bin/env python3
"""Copy or symlink raw YOLO splits into data/processed_data without modifying raw_data."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from dataset_common import get_dataset_entry, load_registry, project_root, resolve_under
from yolo_data_yaml import write_for_registry_dataset

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _iter_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            out.append(p)
    return out


def _validate_label_lines(label_path: Path, nc: int) -> list[str]:
    errors: list[str] = []
    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError as e:
        return [f"{label_path}: read error: {e}"]
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
        try:
            for line in txt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                counts[cid] += 1
        except OSError:
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
    images_sub = entry["images_subdir"]
    labels_sub = entry["labels_subdir"]

    if out_dataset.exists():
        shutil.rmtree(out_dataset)
    out_dataset.mkdir(parents=True, exist_ok=True)

    required_logical = ("train", "val")
    for logical in required_logical:
        if logical not in split_map:
            raise KeyError(f"split_map must include {logical!r}")

    for logical, folder_name in split_map.items():
        src_split = (raw_root / folder_name).resolve()
        dst_split = (out_dataset / folder_name).resolve()
        if not src_split.is_dir():
            if logical in required_logical:
                raise FileNotFoundError(f"Required split missing: {src_split}")
            print(f"Warning: optional split {logical!r} skipped (missing {src_split})")
            continue

        if use_symlinks:
            dst_split.symlink_to(src_split, target_is_directory=True)
        else:
            shutil.copytree(
                src_split,
                dst_split,
                symlinks=False,
                ignore_dangling_symlinks=True,
                dirs_exist_ok=True,
            )

        src_images = src_split / images_sub
        src_labels = src_split / labels_sub
        # After copy, dst has same structure; validate on destination
        dst_images = dst_split / images_sub
        dst_labels = dst_split / labels_sub

        if not use_symlinks and dst_images.is_dir() and dst_labels.is_dir():
            all_errors: list[str] = []
            images = _iter_images(dst_images)
            label_stems = {p.stem for p in dst_labels.glob("*.txt")}
            image_stems = {p.stem for p in images}
            for stem in sorted(image_stems - label_stems):
                print(f"Warning: image without label: {stem} ({logical}/{images_sub})")
            for stem in sorted(label_stems - image_stems):
                print(f"Warning: label without image: {stem} ({logical}/{labels_sub})")
            for lbl in sorted(dst_labels.glob("*.txt")):
                all_errors.extend(_validate_label_lines(lbl, nc))
            if all_errors:
                for msg in all_errors[:50]:
                    print(f"Label issue: {msg}")
                if len(all_errors) > 50:
                    print(f"... {len(all_errors) - 50} more label issues")
                if strict_labels:
                    raise ValueError("Strict label validation failed")
        elif use_symlinks and src_images.is_dir() and src_labels.is_dir():
            # Validate on source when symlinked (read-only raw)
            for lbl in sorted(src_labels.glob("*.txt")):
                errs = _validate_label_lines(lbl, nc)
                for msg in errs[:20]:
                    print(f"Label issue: {msg}")
                if strict_labels and errs:
                    raise ValueError("Strict label validation failed")

        if scan_labels:
            scan_dir = src_labels if use_symlinks else dst_labels
            dist = _scan_label_classes(scan_dir)
            print(f"Class counts in labels ({logical}): {dict(sorted(dist.items()))}")

    if write_yaml:
        yaml_path = write_for_registry_dataset(dataset_id, registry=registry, check_only=False)
        print(f"Wrote data yaml: {yaml_path}")
    return out_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare processed YOLO dataset from raw (no raw writes).")
    ap.add_argument("--dataset", required=True, help="Registry dataset id (e.g. mvp_test)")
    ap.add_argument("--link", action="store_true", help="Symlink split folders instead of copying")
    ap.add_argument("--scan-labels", action="store_true", help="Print class id histograms per split")
    ap.add_argument(
        "--strict-labels",
        action="store_true",
        help="Exit with error if any label line has invalid class id",
    )
    ap.add_argument("--no-yaml", action="store_true", help="Do not write data.yaml after prep")
    args = ap.parse_args()
    prepare_dataset(
        args.dataset,
        use_symlinks=args.link,
        scan_labels=args.scan_labels,
        strict_labels=args.strict_labels,
        write_yaml=not args.no_yaml,
    )


if __name__ == "__main__":
    main()
