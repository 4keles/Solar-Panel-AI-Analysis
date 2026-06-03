"""Training configuration schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    path: str
    train: str
    val: str
    test: str | None = None
    nc: int
    names: list[str] | dict[int, str]
    modality: str | None = None
    version: str | None = None
    notes: str | None = None


class TrainConfig(BaseModel):
    model: str = 'yolo11n.pt'
    data: str
    imgsz: int = Field(default=640, ge=64)
    epochs: int = Field(default=100, ge=1)
    batch: int = 4
    workers: int = 2
    device: str | int = 'cpu'
    half: bool = False
    cos_lr: bool = False
    patience: int = 20
    project: str = 'runs/detect'
    name: str = 'solar_local'
    cache: bool = False

    # Augmentation / loss hyperparams (optional — passed to YOLO only if set)
    hsv_h: float | None = None
    hsv_s: float | None = None
    hsv_v: float | None = None
    degrees: float | None = None
    translate: float | None = None
    scale: float | None = None
    shear: float | None = None
    perspective: float | None = None
    flipud: float | None = None
    fliplr: float | None = None
    mosaic: float | None = None
    mixup: float | None = None
    cls: float | None = None
