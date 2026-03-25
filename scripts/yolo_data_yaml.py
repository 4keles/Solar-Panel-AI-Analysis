#!/usr/bin/env python3
"""Write Ultralytics-compatible data.yaml for a processed dataset layout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from dataset_common import get_dataset_entry, load_registry, project_root, resolve_under


def write_data_yaml(
    *,
    dataset_root: Path,
    out_yaml: Path,
    class_names: list[str],
    train_images_rel: str,
    val_images_rel: str,
    test_images_rel: str | None,
) -> None:
    """Write YOLO data config. Paths train/val/test are relative to dataset_root."""
    nc = len(class_names)
    # 'names' as list keeps compatibility with scripts expecting list-like indexing.
    names: list[str] = list(class_names)
    payload: dict[str, Any] = {
        "path": str(dataset_root.resolve()),
        "train": train_images_rel,
        "val": val_images_rel,
        "nc": nc,
        "names": names,
    }
    if test_images_rel:
        payload["test"] = test_images_rel
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def build_images_relpaths(entry: dict[str, Any]) -> tuple[str, str, str | None]:
    """Return train, val, test relative paths to image dirs (or None if test absent)."""
    split_map = entry["split_map"]
    images_subdir = entry["images_subdir"]
    train_folder = split_map["train"]
    val_folder = split_map["val"]
    train_rel = f"{train_folder}/{images_subdir}".replace("\\", "/")
    val_rel = f"{val_folder}/{images_subdir}".replace("\\", "/")
    test_key = "test"
    if test_key in split_map:
        test_folder = split_map[test_key]
        test_rel = f"{test_folder}/{images_subdir}".replace("\\", "/")
    else:
        test_rel = None
    return train_rel, val_rel, test_rel


def write_for_registry_dataset(
    dataset_id: str,
    *,
    registry: dict[str, Any] | None = None,
    check_only: bool = False,
) -> Path:
    root = project_root()
    reg = registry if registry is not None else load_registry()
    entry = get_dataset_entry(reg, dataset_id)
    processed_root = resolve_under(root, reg["processed_data_root"])
    output_subdir = entry["output_subdir"]
    dataset_root = (processed_root / output_subdir).resolve()
    out_yaml = (processed_root / f"{output_subdir}.data.yaml").resolve()
    class_names = list(entry["class_names"])
    train_rel, val_rel, test_rel = build_images_relpaths(entry)
    if test_rel and not (dataset_root / test_rel).is_dir():
        test_rel = None

    if check_only:
        for name, rel in [("train", train_rel), ("val", val_rel)]:
            p = dataset_root / rel
            if not p.is_dir():
                raise FileNotFoundError(f"Missing split directory for check: {p}")
        if test_rel:
            p = dataset_root / test_rel
            if not p.is_dir():
                raise FileNotFoundError(f"Missing test split directory: {p}")
        return out_yaml

    write_data_yaml(
        dataset_root=dataset_root,
        out_yaml=out_yaml,
        class_names=class_names,
        train_images_rel=train_rel,
        val_images_rel=val_rel,
        test_images_rel=test_rel,
    )
    return out_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Ultralytics data.yaml from dataset_registry.yaml")
    ap.add_argument("--dataset", required=True, help="Dataset id in registry (e.g. mvp_test)")
    ap.add_argument(
        "--check",
        action="store_true",
        help="Only verify processed layout exists; do not write YAML",
    )
    args = ap.parse_args()
    out = write_for_registry_dataset(args.dataset, check_only=args.check)
    if args.check:
        print(f"OK: layout valid for {args.dataset!r} (would write {out})")
    else:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
