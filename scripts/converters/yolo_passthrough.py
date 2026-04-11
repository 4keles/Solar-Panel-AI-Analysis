"""Pass-through converter for already YOLO-formatted datasets."""

from __future__ import annotations

import shutil
from pathlib import Path


def copy_or_link_yolo_splits(
    raw_root: Path,
    processed_root: Path,
    split_map: dict[str, str],
    use_symlinks: bool = False,
) -> Path:
    """Copy/link YOLO split folders to target location."""
    processed_root.mkdir(parents=True, exist_ok=True)
    logical_to_target = {'train': 'train', 'val': 'val', 'test': 'test'}

    for logical, src_name in split_map.items():
        src = raw_root / src_name
        dst = processed_root / logical_to_target.get(logical, logical)
        if not src.is_dir():
            if logical in {'train', 'val'}:
                raise FileNotFoundError(f'Missing required split: {src}')
            continue

        if dst.exists() and not use_symlinks:
            shutil.rmtree(dst)
        if use_symlinks:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src, target_is_directory=True)
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True)

    return processed_root
