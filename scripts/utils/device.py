"""Device helper utilities for local training/inference."""

from __future__ import annotations

from typing import Any


def select_device(prefer_gpu: bool = True) -> str:
    """Select available device identifier for Ultralytics/Torch."""
    try:
        import torch
    except Exception:
        return 'cpu'

    if prefer_gpu and torch.cuda.is_available():
        return 'cuda:0'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def detect_hardware() -> dict[str, Any]:
    """Collect minimal runtime hardware metadata."""
    info: dict[str, Any] = {'device': 'cpu', 'gpu_name': 'unknown', 'vram_gb': 0}
    try:
        import torch

        if torch.cuda.is_available():
            info['device'] = 'cuda:0'
            info['gpu_name'] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            info['vram_gb'] = round(total / (1024 ** 3), 2)
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            info['device'] = 'mps'
    except Exception:
        pass
    return info
