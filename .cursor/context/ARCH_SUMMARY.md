# MİMARİ ÖZET — Token-Verimli Başvuru Kartı
# ============================================================
# Bu dosya architecture.md'nin SIKIŞTIRILMIŞ özetidir.
# Tam architecture.md'yi OKUMA — sadece buradaki bilgiyi kullan.
# Çakışma veya derin mimari soru → CONTRACTS.md'ye git.
# ============================================================

## KLASÖR HARİTASI (Kesin)

```
solar-panel-detection/
├── configs/          ← YAML konfigürasyonlar (kod içinde hardcode YOK)
├── data/
│   ├── raw_data/     ← SALT-OKUNUR, asla yazma
│   └── processed_data/ ← YOLO formatı, train/val/test/
├── models/           ← Versiyonlanmış artifact'lar (v1.0.0/, v1.0.1/, latest→)
├── reports/          ← Versiyonlanmış grafikler ve metrikler
├── runs/             ← YOLO ham çıktı (geçici, .gitignore)
├── scripts/          ← İş mantığı modülleri
│   └── utils/        ← Paylaşılan yardımcılar
└── tests/            ← unit/ ve integration/
```

## MODÜL SÖZLEŞME TABLOSU

| Modül | Alır | Verir | Dokunduğu Dizin |
|---|---|---|---|
| `dataset_prep.py` | `raw_data/` yolu, format, split oranı | YOLO klasör yapısı, `dataset.yaml` | `processed_data/` |
| `augment.py` | `processed_data/` yolu, hedef sınıf, sayı | Augmented görüntü+etiket çiftleri | `processed_data/train/` |
| `train.py` | config YAML, model adı, mod | `runs/` altında eğitim çıktısı | `runs/` |
| `versioning.py` | `runs/` çıktı yolu, bump türü | `models/vX.Y.Z/` dizini | `models/` |
| `metadata.py` | eğitim sonuçları, args | `metadata.json` | `models/vX.Y.Z/` |
| `validate.py` | `.pt` yolu, data yaml, split | metrik dict, grafikler | `reports/vX.Y.Z/` |
| `predict_live.py` | `.pt` yolu, source | ekran/video/log | `output/` |
| `export_model.py` | `.pt` yolu, format | dönüştürülmüş model | `models/vX.Y.Z/exports/` |
| `benchmark.py` | `.pt` yolu, imgsz, device | FPS/latency raporu | stdout + `reports/` |

## KRİTİK VERİ TİPLERİ

```python
# Her modülün kabul ettiği/ürettiği temel tipler:

ModelVersion = str          # "v1.0.1" formatı, SemVer
DeployMode = Literal["edge", "host"]
Modality = Literal["rgb", "thermal", "fusion"]
BumpType = Literal["major", "minor", "patch"]
DeviceSpec = str            # "cpu" | "cuda:0" | "mps" | "0,1"

# metadata.json'ın Python karşılığı → scripts/utils/metadata.py'de tanımlı
```

## BAĞIMLILIK YÖNÜ (Oklar tek yönlü)

```
configs/  ──────────────────────────────→ tüm script'ler okur
data/raw_data/  ──→  dataset_prep.py  ──→  data/processed_data/
data/processed_data/  ──→  train.py  ──→  runs/
runs/  ──→  versioning.py  ──→  models/
models/  ──→  validate.py  ──→  reports/
models/  ──→  predict_live.py  ──→  output/
models/  ──→  export_model.py  ──→  models/exports/
```

**KURAL:** Yukarı yönde bağımlılık YASAK (ör: `train.py` → `validate.py` çağıramaz)

## SINIF TANIMLAMALARI (dataset.yaml)

```
0: crack    1: dirt    2: shadow    3: hotspot
```

## VERSİYON ANLAMI

```
vMAJOR.MINOR.PATCH
  │      │     └── Hiperparametre/küçük fix
  │      └────── Yeni sınıf veya büyük veri seti değişikliği
  └────────── Modalite değişikliği (RGB→Thermal) veya mimari değişiklik
```

## DEPLOYMENT MOD FARKLARI

| Parametre | Edge | Host |
|---|---|---|
| Model | yolo11n/s | yolo11m/l/x |
| imgsz | 416 | 1280 |
| batch | -1 (auto) | -1 (auto) |
| half | false (RPi) / true (Jetson) | true |
| Export hedefi | tflite/onnx | engine (TensorRT) |

## LOGLAMA KURALI

```python
# Her modülde:
from scripts.utils.logger import get_logger
logger = get_logger(__name__)

# Asla print() kullanma
logger.info("event_name", key=value, key2=value2)
```
