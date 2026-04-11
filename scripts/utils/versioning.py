"""Model artifact versioning helpers."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Literal

BumpType = Literal['major', 'minor', 'patch']


def _parse_version(version: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def list_versions(models_dir: Path) -> list[str]:
    """Return SemVer-like model version directories sorted ascending."""
    versions: list[str] = []
    if not models_dir.exists():
        return versions
    for d in models_dir.iterdir():
        if d.is_dir() and re.fullmatch(r"v\d+\.\d+\.\d+", d.name):
            versions.append(d.name)
    return sorted(versions, key=_parse_version)


def get_next_version(models_dir: Path, bump: BumpType) -> str:
    """Calculate next version using semantic bump strategy."""
    versions = list_versions(models_dir)
    if not versions:
        return 'v1.0.0'

    major, minor, patch = _parse_version(versions[-1])
    if bump == 'major':
        return f'v{major + 1}.0.0'
    if bump == 'minor':
        return f'v{major}.{minor + 1}.0'
    if bump == 'patch':
        return f'v{major}.{minor}.{patch + 1}'
    raise ValueError(f'Unsupported bump type: {bump}')


def update_latest_symlink(models_dir: Path, version: str) -> None:
    """Update models/latest symlink to target version folder."""
    target = models_dir / version
    target.mkdir(parents=True, exist_ok=True)

    latest = models_dir / 'latest'
    if latest.is_symlink() or latest.exists():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    latest.symlink_to(version)


def promote_run_artifacts(run_dir: Path, version: str, models_dir: Path) -> Path:
    """Copy run artifacts to models/version folder and update latest."""
    src_weights = run_dir / 'weights'
    if not src_weights.is_dir():
        raise FileNotFoundError(f'Missing run weights directory: {src_weights}')

    target = models_dir / version
    target.mkdir(parents=True, exist_ok=True)

    for name in ('best.pt', 'last.pt'):
        src = src_weights / name
        if src.exists():
            shutil.copy2(src, target / name)

    update_latest_symlink(models_dir, version)
    return target
