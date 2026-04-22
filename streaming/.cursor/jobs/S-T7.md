# JOB: S-T7 — Config, CLI ve main.py
# ============================================================
# BAĞIMLILIK: Yok (S-T1, S-T3 ile paralel)
# NOT: main.py İSKELETİ S-T7'de, tam implementasyon S-T6 sonra
# ============================================================

## AGENT PROMPT

> Sen bir Kıdemli Python Mühendisisin. Bu JOB konfigürasyon sistemi ve komut satırı arayüzünü kurar. `CONTRACTS.md` → `SÖZLEŞME S-006`'yı oku. Ana projenin `config_loader.py` kullanılır — kopyalama.

## GÖREV TANIMI

3 çıktı üreteceksin:
1. `streaming/configs/streaming.yaml` — Ana konfigürasyon şablonu
2. `streaming/configs/display.yaml` — Görsel overlay ayarları
3. `streaming/main.py` — Entry point + CLI

## INPUTS

```
READS:
  - streaming/.cursor/context/CONTRACTS.md  (SÖZLEŞME S-006)
  - streaming/.cursor/context/ARCH_SUMMARY.md
```

## OUTPUTS

```
WRITES:
  - streaming/configs/streaming.yaml
  - streaming/configs/display.yaml
  - streaming/main.py
  - streaming/tests/unit/test_config.py
```

## UYGULAMA REHBERİ

### streaming.yaml (Kesin Şema — SÖZLEŞME S-006)

```yaml
model:
  path: ""                    # Boş = ../models/latest/best.pt dener
  modality: "rgb"             # "rgb" | "thermal"
  conf: 0.25
  iou: 0.45
  device: "0"                 # "cpu" | "0"

source:
  type: "camera"              # "camera" | "video" | "rtsp"
  camera_id: 0
  video_path: ""
  rtsp_url: ""

recording:
  enabled: false
  output_dir: "output/recordings"
  codec: "mp4v"
  fps_override: null

display:
  show_window: true
  window_title: "Solar Panel — Detection"
  scale: 1.0
  show_fps: true
  show_conf: true
  show_class_label: true
```

### display.yaml

```yaml
bbox:
  thickness: 2
  font_scale: 0.6
  font: "FONT_HERSHEY_SIMPLEX"
  label_padding: 4

hud:
  fps_position: [10, 30]       # x, y
  source_position: [10, 60]
  rec_indicator_offset: [-120, 30]  # sağdan
  text_color: [255, 255, 255]
  background_alpha: 0.4
```

### main.py Yapısı

```python
#!/usr/bin/env python3
"""Solar Panel OD — Live Streaming Entry Point."""

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solar Panel Detection — Streaming")
    p.add_argument("--config", type=Path, default=Path("configs/streaming.yaml"))
    p.add_argument("--source", help="Override: 0, video.mp4, rtsp://...")
    p.add_argument("--model", type=Path, help="Override: path to .pt")
    p.add_argument("--record", action="store_true", help="Start recording immediately")
    p.add_argument("--no-gui", action="store_true", help="Headless mode (record only)")
    p.add_argument("--device", default=None, help="Override: cpu | 0 | cuda:0")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--dry-run", action="store_true", help="Load config only, don't start")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    # 1. Config yükle (ana proj config_loader kullan)
    # 2. CLI override'ları uygula
    # 3. --dry-run → config çıktısı bas, çık
    # 4. UI modunu belirle (GUI vs headless)
    # 5. Pipeline başlat
    ...

if __name__ == "__main__":
    main()
```

### CLI Kullanım Örnekleri (README için de kullanılır)

```bash
# Webcam, gerçek zamanlı, GUI ile:
python main.py --source 0 --model ../models/latest/best.pt

# Video dosyası, kayıt açık:
python main.py --source footage.mp4 --record

# Headless mode (sunucu/edge):
python main.py --source rtsp://192.168.1.100/stream --no-gui --record

# Config dosyasını override ile:
python main.py --config configs/streaming.yaml --device cpu --conf 0.30
```

## SELF_TEST

```bash
cd streaming
python main.py --dry-run   # Config parse edilir, çıkar
python main.py --dry-run --source 0 --model ../yolo11n.pt  # Override test

python -m pytest tests/unit/test_config.py -v
```

## TEST GEREKSİNİMLERİ

- `test_default_config_loads`: YAML başarıyla parse edilir
- `test_cli_source_override`: `--source 0` → config.source.camera_id=0
- `test_cli_model_override`: `--model x.pt` → config.model.path güncellenir
- `test_dry_run_exits_clean`: `sys.exit` çağrılmaz, çıktı basılır
- `test_no_gui_sets_headless`: `args.no_gui=True` headless moda girer

## TAMAMLAMA PROTOKOLÜ

1. `python main.py --dry-run` → hatasız çalışır
2. Testler PASS
3. `PROJECT_STATE.md` → S-T7: ✅ DONE
4. `git commit -m "S-T7: Config & CLI & main.py implemented ✅"`
