#!/usr/bin/env python3
"""Write Ultralytics-compatible data yaml for a processed dataset layout."""

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
    payload: dict[str, Any] = {
        'path': str(dataset_root),
        'train': train_images_rel,
        'val': val_images_rel,
        'nc': len(class_names),
        'names': list(class_names),
    }
    if test_images_rel:
        payload['test'] = test_images_rel
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.dump(payload, sort_keys=False, allow_unicode=True), encoding='utf-8')


def build_images_relpaths(entry: dict[str, Any]) -> tuple[str, str, str | None]:
    images_subdir = entry.get('images_subdir', 'images')
    train_rel = f'train/{images_subdir}'
    val_rel = f'val/{images_subdir}'
    test_rel = f'test/{images_subdir}' if 'test' in entry.get('split_map', {}) else None
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

    dataset_root = resolve_under(root, f"{reg['processed_data_root']}/{entry['output_subdir']}")
    out_yaml = resolve_under(root, f"{reg['processed_data_root']}/{entry['output_subdir'].replace('/', '_')}.data.yaml")

    train_rel, val_rel, test_rel = build_images_relpaths(entry)
    if check_only:
        for rel in (train_rel, val_rel):
            if not (dataset_root / rel).is_dir():
                raise FileNotFoundError(f'Missing split: {dataset_root / rel}')

    write_data_yaml(
        dataset_root=dataset_root,
        out_yaml=out_yaml,
        class_names=list(entry['class_names']),
        train_images_rel=train_rel,
        val_images_rel=val_rel,
        test_images_rel=test_rel,
    )
    return out_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Ultralytics data yaml from registry')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    out = write_for_registry_dataset(args.dataset, check_only=args.check)
    print(out)


if __name__ == '__main__':
    main()
