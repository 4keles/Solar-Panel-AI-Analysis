# Solar Panel Defect Detection — YOLO11 Multi-Modal

> End-to-end solar panel anomaly detection: 6-class fault classification from RGB and thermal imagery, with a PyQt6 live-stream dashboard and an active learning annotation pipeline.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Model-4keles%2Fsolar--panel--od-yellow)](https://huggingface.co/4keles/solar-panel-od)
[![YOLO](https://img.shields.io/badge/YOLO-11-red)](https://github.com/ultralytics/ultralytics)

---

## Demo

> Screenshot / GIF not yet added. To generate: run `python main.py`, open a solar panel video in the dashboard, and capture a frame. PR welcome.

---

## Features

- **6-class fault detection** — physical damage, dust particles, bird drops, bird feathers, leaf, snow
- **Multi-modal support** — RGB and thermal camera input with live mode toggle and thermal contrast/brightness controls
- **ONNX + TensorRT export** — ONNX for CPU/GPU portability (37.9 MB); TensorRT `.engine` for edge deployment
- **PyQt6 live-stream dashboard** — multi-source input (USB cam, IP cam, RTSP/RTMP, MP4), ByteTrack object tracking, clean recording without OSD overlay
- **Active Learning pipeline** — Label Studio integration for human-in-the-loop annotation from drone footage
- **Modular design** — inference, streaming, training, and annotation pipelines are independent and composable

---

## Architecture

```
Input (RGB / Thermal camera, RTSP, MP4)
            │
            ▼
    Pre-processing  ◄── configs/
            │
            ▼
  YOLO11 Inference ──── models/v1.2.1/best.onnx  (37.9 MB)
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
  PyQt6         scripts/
  Dashboard     (train / augment / active-learn / export)
     │
     ▼
  output/  (recordings, snapshots — no OSD overlay)
```

---

## Installation

**Requirements:** Python 3.10+, CUDA 12.4, [`uv`](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/4keles/Solar-Panel-AI-Analysis.git
cd Solar-Panel-AI-Analysis

uv sync
```

**Environment variables:**

```bash
cp .env.example .env
# Edit .env — set HF_TOKEN if downloading from a private HF repo,
# set LABEL_STUDIO_URL if using the active learning pipeline
```

**TensorRT (optional):** `uv sync` installs the Python bindings (`tensorrt-cu12`), but the runtime must be installed system-wide. Follow the [NVIDIA TensorRT install guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) for CUDA 12.4.

---

## Quick Start

### 1 — Download the model

```bash
python scripts/download_model.py --version v1.2.1 --format onnx
# Downloads best.onnx (37.9 MB) into models/v1.2.1/
```

All model versions are hosted on [HuggingFace Hub](https://huggingface.co/4keles/solar-panel-od).

### 2 — Launch the dashboard

```bash
python main.py
```

Select a source (webcam, MP4, or RTSP URL) and model version from the sidebar. Switch to Thermal mode for infrared input.

### 3 — Headless inference (scripted)

```python
from ultralytics import YOLO

model = YOLO("models/v1.2.1/best.onnx")
results = model.predict("path/to/image.jpg", conf=0.25)
results[0].show()
```

### 4 — Export to TensorRT

```bash
python scripts/export_engine.py --model models/v1.2.1/best.pt --imgsz 640
# Outputs models/v1.2.1/best.engine (CUDA 12.4 specific)
```

---

## Model Performance — v1.2.1

Evaluated on held-out test split. Source: `reports/v1.2.1/val_summary.json`.

| Class | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| **Overall** | **0.546** | **0.241** | **0.569** | **0.582** |
| bird_feather | 0.995 | 0.498 | 0.832 | 1.000 |
| leaf | 0.752 | 0.302 | 0.668 | 0.813 |
| physical_damage | 0.552 | 0.251 | 0.543 | 0.565 |
| snow | 0.467 | 0.202 | 0.567 | 0.494 |
| dust_partical | 0.408 | 0.160 | 0.590 | 0.373 |
| bird_drop | 0.100 | 0.030 | 0.214 | 0.246 |

All versions with model cards: [huggingface.co/4keles/solar-panel-od](https://huggingface.co/4keles/solar-panel-od)

---

## Developer Guide

### Active Learning (human-in-the-loop annotation)

```bash
# Start Label Studio in a separate terminal
label-studio start

# Run the auto-annotation pipeline on raw drone footage
python scripts/active_learning_pipeline.py \
  --image-dir data/raw_data/unlabeled \
  --model models/v1.2.1/best.pt
```

### Training

```bash
python scripts/train.py --config scripts/schemas/train_config.yaml
```

### Data augmentation

```bash
python scripts/augment.py \
  --source data/processed_data/rgb_master/train \
  --target-count 5000
```

### Run tests

```bash
pytest tests/
```

---

## Project Structure

```
solar_panel_od/
├── configs/              # YAML configs for training and UI
├── data/                 # Datasets and labels (gitignored — ~25 GB)
├── docs/                 # Active learning guide and research notes
├── models/               # Model weights — downloaded via download_model.py
│   └── v1.2.1/           # best.pt (PyTorch) + best.onnx (37.9 MB)
├── reports/              # Per-version val_summary.json
│   └── v1.2.1/
├── scripts/              # Training, augmentation, export, download utilities
│   ├── download_model.py # Pull weights from HuggingFace Hub
│   ├── train.py
│   ├── augment.py
│   ├── export_engine.py  # ONNX → TensorRT
│   └── active_learning_pipeline.py
├── streaming/            # PyQt6 live dashboard source
├── tests/                # Pytest test suite
├── tools/                # Utility scripts
├── main.py               # Dashboard entry point
├── pyproject.toml        # Dependencies (uv)
└── .env.example          # Env variable template
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{solar_panel_od_2026,
  author    = {4keles},
  title     = {Solar Panel Defect Detection — YOLO11 Multi-Modal},
  year      = {2026},
  url       = {https://github.com/4keles/Solar-Panel-AI-Analysis},
  note      = {Model weights: https://huggingface.co/4keles/solar-panel-od}
}
```
