"""Training metadata models and helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from .device import detect_hardware
except Exception:  # pragma: no cover
    from scripts.utils.device import detect_hardware

DeployMode = Literal['edge', 'host', 'local']
Modality = Literal['rgb', 'thermal', 'fusion']


class DatasetInfo(BaseModel):
    path: str
    source: str = 'unknown'
    notes: str = 'unknown'


class HyperParams(BaseModel):
    model: str
    epochs: int
    imgsz: int
    batch: int
    device: str


class TrainingResults(BaseModel):
    precision: float | None = None
    recall: float | None = None
    mAP50: float | None = None
    mAP50_95: float | None = Field(default=None, alias='mAP50_95')


class HardwareInfo(BaseModel):
    gpu: str = 'unknown'
    vram_gb: float = 0
    ram_gb: int | str = 'unknown'
    cpu: str = 'unknown'


class GitInfo(BaseModel):
    commit: str = 'unknown'
    branch: str = 'unknown'
    dirty: bool = False


class TrainingMetadata(BaseModel):
    version: str
    created_at: datetime
    base_model: str
    deployment_mode: DeployMode
    data_modality: Modality
    dataset: DatasetInfo
    hyperparameters: HyperParams
    results: TrainingResults | None = None
    hardware: HardwareInfo
    git: GitInfo
    resumed_from: str | None = None
    finetuned_from: str | None = None


def build_training_metadata(version: str, args: Any, results: dict[str, Any] | None, run_dir: Path, git_info: dict[str, Any]) -> TrainingMetadata:
    hw = detect_hardware()
    return TrainingMetadata(
        version=version,
        created_at=datetime.now(timezone.utc),
        base_model=str(getattr(args, 'model', 'yolo11n.pt')),
        deployment_mode=getattr(args, 'mode', 'local'),
        data_modality='rgb',
        dataset=DatasetInfo(path=str(getattr(args, 'data_config', 'configs/dataset_rgb.yaml')),
                           source='mvp_test_v1',
                           notes='Etiketli ilk MVP veri seti'),
        hyperparameters=HyperParams(
            model=str(getattr(args, 'model', 'yolo11n.pt')),
            epochs=int(getattr(args, 'epochs', 0) or 0),
            imgsz=int(getattr(args, 'imgsz', 0) or 0),
            batch=int(getattr(args, 'batch', 0) or 0),
            device=str(getattr(args, 'device', hw.get('device', 'cpu'))),
        ),
        results=TrainingResults.model_validate(results or {}),
        hardware=HardwareInfo(gpu=str(hw.get('gpu_name', 'unknown')),
                              vram_gb=float(hw.get('vram_gb', 0)),
                              cpu='unknown'),
        git=GitInfo(**git_info),
        resumed_from=getattr(args, 'resume', None),
        finetuned_from=getattr(args, 'finetune', None),
    )


def write_metadata(path: Path, data: TrainingMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data.model_dump_json(indent=2), encoding='utf-8')


def read_metadata(version_dir: Path) -> TrainingMetadata:
    return TrainingMetadata.model_validate_json((version_dir / 'metadata.json').read_text(encoding='utf-8'))
