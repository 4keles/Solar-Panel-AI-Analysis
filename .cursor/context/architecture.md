# Solar Panel Object Detection — System Architecture

> **Versiyon:** 1.0.0  
> **Model Ailesi:** YOLOv11  
> **Python:** ≥ 3.10  
> **Paket Yöneticisi:** `uv`  
> **Tasarım Prensibi:** Divide & Conquer — Yüksek Derecede Modüler Mimari  
> **Mevcut Veri Modalitesi:** RGB (Termal entegrasyonu için genişletilebilir)

---

## İçindekiler

1. [Genel Bakış ve Tasarım Felsefesi](#1-genel-bakış-ve-tasarım-felsefesi)
2. [Klasör ve Dosya Hiyerarşisi](#2-klasör-ve-dosya-hiyerarşisi)
3. [Ortam Kurulumu (uv)](#3-ortam-kurulumu-uv)
4. [Veri Katmanı](#4-veri-katmanı)
5. [Model Katmanı ve Versiyonlama](#5-model-katmanı-ve-versiyonlama)
6. [Çift Modlu Dağıtım Stratejisi](#6-çift-modlu-dağıtım-stratejisi)
7. [Gelişmiş Eğitim Boru Hattı](#7-gelişmiş-eğitim-boru-hattı)
8. [Doğrulama ve Raporlama](#8-doğrulama-ve-raporlama)
9. [Canlı Çıkarım (Live Inference)](#9-canlı-çıkarım-live-inference)
10. [Yapılandırma Yönetimi](#10-yapılandırma-yönetimi)
11. [CI/CD ve Kalite Kapıları](#11-cicd-ve-kalite-kapıları)
12. [Gözlemlenebilirlik ve Loglama](#12-gözlemlenebilirlik-ve-loglama)
13. [Güvenlik ve Erişim Kontrolü](#13-güvenlik-ve-erişim-kontrolü)
14. [Termal Veri Entegrasyonu (Gelecek)](#14-termal-veri-entegrasyonu-gelecek)
15. [Mimari Karar Kayıtları (ADR)](#15-mimari-karar-kayıtları-adr)

---

## 1. Genel Bakış ve Tasarım Felsefesi

### 1.1 Proje Amacı

Bu sistem, güneş panellerindeki fiziksel ve elektriksel arızaları (çatlaklar, kir, gölgelenme, bağlantı kopukluğu vb.) RGB görüntüler üzerinden nesne tespiti yöntemiyle otomatik olarak tespit etmek için tasarlanmıştır. Sistem, uçtan uca bir ML yaşam döngüsünü destekler: ham veri alımından canlı dağıtıma kadar tüm süreçler modüler, versiyonlanmış ve yeniden üretilebilir biçimde yönetilir.

### 1.2 Temel Tasarım Prensipleri

| Prensip | Açıklama |
|---|---|
| **Divide & Conquer** | Her iş birimi tek bir sorumluluk alanına sahip ayrı bir modüldür. |
| **Yeniden Üretilebilirlik** | Tüm eğitim deneyleri `metadata.json` ve sabit seed'ler aracılığıyla tam olarak tekrarlanabilir. |
| **Kademeli Yükseltme** | Nano modelden büyük modele, RGB'den Termale doğru kesintisiz geçiş yapılabilir. |
| **Başarısızlık Direnci** | `--resume` ile eğitim kurtarma, `--dry-run` ile boru hattı doğrulama desteklenir. |
| **Gözlemlenebilirlik** | Her aşama yapılandırılmış log, metrik ve artifact üretir. |
| **Sıfır Gizli Durum** | Tüm konfigürasyon YAML/JSON dosyalarında açıkça saklanır; ortam değişkenlerine kritik bağımlılık yoktur. |

### 1.3 Sistem Bileşen Diyagramı

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SOLAR PANEL DETECTION SYSTEM                    │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  DATA LAYER  │───▶│ TRAIN LAYER  │───▶│   ARTIFACT LAYER         │  │
│  │              │    │              │    │                          │  │
│  │  raw_data/   │    │  train.py    │    │  models/v1.0.0/          │  │
│  │  processed/  │    │  augment.py  │    │  ├── best.pt             │  │
│  │  dataset.yaml│    │  callbacks/  │    │  └── metadata.json       │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│          │                                           │                  │
│          ▼                                           ▼                  │
│  ┌──────────────┐                        ┌──────────────────────────┐  │
│  │  PREP LAYER  │                        │   VALIDATION LAYER       │  │
│  │              │                        │                          │  │
│  │  dataset_    │                        │  validate.py             │  │
│  │  prep.py     │                        │  reports/v1.0.0/         │  │
│  └──────────────┘                        └──────────────────────────┘  │
│                                                      │                  │
│                                                      ▼                  │
│                                           ┌──────────────────────────┐ │
│                              ┌──────────▶ │   INFERENCE LAYER        │ │
│                              │            │                          │ │
│                   ┌──────────┴──────────┐ │  predict_live.py         │ │
│                   │  DEPLOYMENT MODE    │ │  ├── Edge  (nano/small)  │ │
│                   │  (Config-driven)    │ │  └── Host  (medium/large)│ │
│                   └─────────────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Klasör ve Dosya Hiyerarşisi

Aşağıdaki yapı, Divide & Conquer prensibine göre her sorumluluğu kendi ad alanında izole eder. Hiçbir modül, kendi katmanının dışındaki verilere doğrudan yazmaz.

```
solar-panel-detection/
│
├── .python-version                  # uv tarafından yönetilen Python versiyonu (örn: 3.11)
├── pyproject.toml                   # Proje meta verisi, bağımlılıklar, araç konfigürasyonları
├── uv.lock                          # Kilitlenmiş bağımlılık ağacı (commit edilmeli)
├── .env.example                     # Ortam değişkeni şablonu (asla .env commit edilmez)
├── .gitignore
├── README.md
├── architecture.md                  # Bu dosya
│
├── configs/                         # Tüm statik konfigürasyonlar
│   ├── dataset.yaml                 # YOLO veri seti tanım dosyası (nc, names, paths)
│   ├── train_edge.yaml              # Edge modu eğitim hiperparametreleri
│   ├── train_host.yaml              # Host modu eğitim hiperparametreleri
│   └── inference.yaml               # Çıkarım eşik değerleri, NMS parametreleri
│
├── data/
│   ├── raw_data/                    # HAM VERİ — DOKUNULMAZ, salt-okunur pratiği
│   │   ├── rgb/                     # RGB görüntüler (drone, sabit kamera vb.)
│   │   │   ├── images/
│   │   │   └── annotations/         # Orijinal etiket formatı (COCO JSON, Pascal VOC vb.)
│   │   └── thermal/                 # [GELECEK] Termal görüntüler — henüz aktif değil
│   │       ├── images/
│   │       └── annotations/
│   │
│   └── processed_data/              # İşlenmiş, eğitime hazır YOLO formatı
│       ├── rgb/
│       │   ├── train/
│       │   │   ├── images/          # .jpg / .png
│       │   │   └── labels/          # .txt (YOLO normalize bbox)
│       │   ├── val/
│       │   │   ├── images/
│       │   │   └── labels/
│       │   └── test/
│       │       ├── images/
│       │       └── labels/
│       └── thermal/                 # [GELECEK] Aynı yapı termal için
│
├── models/                          # Versiyonlanmış model artifact'ları
│   ├── v1.0.0/
│   │   ├── best.pt                  # En iyi doğrulama ağırlıkları
│   │   ├── last.pt                  # Son epoch ağırlıkları (kurtarma için)
│   │   └── metadata.json            # Eğitim meta verisi (aşağıda şema var)
│   ├── v1.0.1/
│   │   ├── best.pt
│   │   ├── last.pt
│   │   └── metadata.json
│   └── latest -> v1.0.1/            # Sembolik link, her zaman en güncel versiyona işaret eder
│
├── reports/                         # Versiyonlanmış doğrulama çıktıları
│   ├── v1.0.0/
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix_normalized.png
│   │   ├── PR_curve.png
│   │   ├── F1_curve.png
│   │   ├── results.csv              # Epoch bazlı eğitim metrikleri
│   │   └── val_summary.json         # mAP50, mAP50-95, Precision, Recall
│   └── v1.0.1/
│       └── ...
│
├── runs/                            # YOLO'nun ham çıktı dizini (geçici, .gitignore'da)
│   └── detect/
│       └── train_<timestamp>/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           └── ...
│
├── scripts/                         # İş mantığı modülleri
│   ├── __init__.py
│   ├── dataset_prep.py              # Ham veriyi YOLO formatına dönüştürme, bölme
│   ├── augment.py                   # Veri artırma pipeline'ı (Albumentations entegrasyonu)
│   ├── train.py                     # Ana eğitim modülü
│   ├── validate.py                  # Model doğrulama ve rapor üretme
│   ├── predict_live.py              # Gerçek zamanlı çıkarım (kamera / RTSP / dosya)
│   ├── export_model.py              # ONNX, TensorRT, TFLite dönüştürme
│   ├── benchmark.py                 # FPS, latency, bellek tüketimi ölçümü
│   └── utils/
│       ├── __init__.py
│       ├── versioning.py            # Otomatik versiyon hesaplama ve klasör yönetimi
│       ├── metadata.py              # metadata.json okuma/yazma yardımcıları
│       ├── logger.py                # Yapılandırılmış JSON loglama (structlog tabanlı)
│       ├── config_loader.py         # YAML/JSON konfigürasyon yükleyici ve doğrulayıcı
│       └── device.py                # CUDA/MPS/CPU cihaz algılama ve öneri mantığı
│
├── tests/                           # Birim ve entegrasyon testleri
│   ├── conftest.py
│   ├── test_dataset_prep.py
│   ├── test_versioning.py
│   ├── test_metadata.py
│   └── test_predict_live.py
│
├── notebooks/                       # Keşifsel analiz (üretim kodu değil)
│   ├── 01_data_exploration.ipynb
│   ├── 02_augmentation_preview.ipynb
│   └── 03_results_analysis.ipynb
│
└── .github/
    └── workflows/
        ├── lint_and_test.yml        # Her PR'da çalışır
        └── model_validation.yml     # Yeni model versiyonu tag'lendiğinde çalışır
```

### 2.1 metadata.json Şeması

Her eğitim versiyonunun tekrar üretilebilirliğini garanti eden zorunlu alan tanımı:

```json
{
  "version": "v1.0.1",
  "created_at": "2025-01-15T10:32:00Z",
  "base_model": "yolo11n.pt",
  "deployment_mode": "edge",
  "data_modality": "rgb",
  "dataset": {
    "path": "configs/dataset.yaml",
    "num_classes": 4,
    "class_names": ["crack", "dirt", "shadow", "hotspot"],
    "train_images": 3200,
    "val_images": 400,
    "test_images": 400,
    "sha256": "a3f8c2..."
  },
  "hyperparameters": {
    "epochs": 150,
    "imgsz": 640,
    "batch": -1,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "cos_lr": true,
    "focal_loss": true,
    "freeze_layers": 0,
    "augmentation": "auto"
  },
  "results": {
    "mAP50": 0.847,
    "mAP50_95": 0.631,
    "precision": 0.873,
    "recall": 0.812,
    "f1": 0.841,
    "epochs_trained": 150,
    "best_epoch": 137
  },
  "hardware": {
    "device": "cuda:0",
    "gpu_name": "NVIDIA RTX 3090",
    "vram_used_gb": 6.2,
    "training_time_minutes": 87
  },
  "git": {
    "commit": "4a7f2bc",
    "branch": "main",
    "dirty": false
  },
  "resumed_from": null,
  "finetuned_from": null
}
```

---

## 3. Ortam Kurulumu (uv)

`uv`, bu projede hem Python sürümü yönetimi hem de paket yönetimi için tek araç olarak kullanılır. `pip`, `venv` veya `conda` kullanılmaz.

### 3.1 İlk Kurulum

```bash
# uv'yi kur (sistem genelinde, bir kez)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Projeyi klonla
git clone https://github.com/your-org/solar-panel-detection.git
cd solar-panel-detection

# Python versiyonunu sabitle ve sanal ortamı oluştur
uv python install 3.11
uv venv --python 3.11

# Bağımlılıkları kilitli dosyadan tam olarak yükle
uv sync

# Ortamı aktive et
source .venv/bin/activate   # Linux/macOS
# veya
.venv\Scripts\activate      # Windows
```

### 3.2 pyproject.toml Yapısı

```toml
[project]
name = "solar-panel-detection"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "ultralytics>=8.3.0",
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "opencv-python-headless>=4.9.0",
    "albumentations>=1.4.0",
    "structlog>=24.1.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.2.0",
    "rich>=13.7.0",
    "typer>=0.9.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.1",
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.3.0",
    "mypy>=1.9.0",
    "pre-commit>=3.6.0",
]
export = [
    "onnx>=1.16.0",
    "onnxruntime>=1.17.0",
    "onnxsim>=0.4.36",
]
thermal = [
    "pyflir>=0.3.0",           # [GELECEK] Termal kamera entegrasyonu
    "flirpy>=0.5.0",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=scripts --cov-report=term-missing"
```

---

## 4. Veri Katmanı

### 4.1 dataset_prep.py — Veri Hazırlama Modülü

Bu modülün tek sorumluluğu: ham veriyi `raw_data/`'dan alıp YOLO formatına dönüştürerek `processed_data/`'ya yazmaktır. Ham veriye **asla doğrudan yazılmaz**.

```
Desteklenen kaynak formatları:
  - COCO JSON (.json)
  - Pascal VOC (.xml)
  - LabelMe JSON (.json)
  - CVAT for images (.xml)
  - Roboflow export (YOLO formatı doğrudan)
```

**Kullanım:**

```bash
# COCO formatından dönüştür, %80/%10/%10 böl
uv run scripts/dataset_prep.py \
  --source data/raw_data/rgb/ \
  --format coco \
  --output data/processed_data/rgb/ \
  --split 0.8 0.1 0.1 \
  --seed 42 \
  --modality rgb

# Sınıf dağılım raporunu görüntüle
uv run scripts/dataset_prep.py --source data/raw_data/rgb/ --report-only
```

**Modül İç Mimarisi:**

```python
# scripts/dataset_prep.py — Sorumluluk sınırı
class DatasetPreparer:
    """
    Sorumluluklar:
    - Format dönüştürme (COCO/VOC/LabelMe → YOLO)
    - Sınıf indeksi normalizasyonu
    - Stratified train/val/test bölme
    - SHA-256 tabanlı veri parmak izi hesaplama
    - configs/dataset.yaml otomatik üretimi
    - Sınıf dengesizlik raporlama

    Sorumluluk DIŞI (delegate edilir):
    - Augmentation (augment.py)
    - Eğitim (train.py)
    """
```

### 4.2 configs/dataset.yaml

```yaml
# YOLO standart veri seti konfigürasyonu
path: data/processed_data/rgb    # Proje köküne göreli
train: train/images
val:   val/images
test:  test/images

nc: 4
names:
  0: crack          # Panel yüzeyindeki çatlaklar
  1: dirt           # Kir ve toz birikimi
  2: shadow         # Dış gölgelenme (yapı, ağaç vb.)
  3: hotspot        # Termal hotspot iz bölgesi (RGB'de görsel belirteçlerle)

# [GELECEK] Termal modalite için ek alan
# thermal_path: data/processed_data/thermal
# modality: rgb  # rgb | thermal | fusion
```

### 4.3 augment.py — Augmentation Stratejisi

Augmentation, eğitim sırasında YOLO'nun yerleşik pipeline'ına ek olarak, özellikle azınlık sınıfları için ön-augmentation (offline) destekler.

```bash
# Azınlık sınıfı 'hotspot' için offline augmentation
uv run scripts/augment.py \
  --source data/processed_data/rgb/train/ \
  --target-class 3 \
  --target-count 500 \
  --output data/processed_data/rgb/train/ \
  --pipeline configs/augmentation_pipeline.yaml
```

**Augmentation Pipeline (configs/augmentation_pipeline.yaml):**

```yaml
# Güneş paneli görüntüleri için optimize edilmiş augmentation
transforms:
  - RandomBrightnessContrast:      # Aydınlatma varyasyonları
      brightness_limit: 0.3
      contrast_limit: 0.3
      p: 0.7
  - RandomShadow:                  # Gerçekçi gölge simülasyonu
      p: 0.4
  - GaussNoise:                    # Sensör gürültüsü
      var_limit: [10, 50]
      p: 0.3
  - HorizontalFlip:                p: 0.5
  - VerticalFlip:                  p: 0.3
  - RandomRotate90:                p: 0.4
  - Perspective:                   # Drone açısı değişimi
      scale: [0.05, 0.1]
      p: 0.3
  - CLAHE:                         # Düşük kontrast paneller
      p: 0.2
  - ImageCompression:              # RTSP stream artefaktları
      quality_lower: 75
      p: 0.2
```

---

## 5. Model Katmanı ve Versiyonlama

### 5.1 Versiyon Numaralandırma Stratejisi

Semantic Versioning (SemVer) prensibine uyarlanmış model versiyonlaması:

```
v<MAJOR>.<MINOR>.<PATCH>

MAJOR: Veri modalitesi veya mimari değişikliği (RGB→Thermal, YOLOv11→v12)
MINOR: Yeni sınıf eklenmesi veya kapsamlı veri seti güncellemesi
PATCH: Hiperparametre ayarı, ince ayar (fine-tuning), hata düzeltmesi
```

**Örnekler:**

| Versiyon | Açıklama |
|---|---|
| `v1.0.0` | İlk RGB modeli, 4 sınıf, yolo11n |
| `v1.0.1` | Aynı veri, lr ve epoch düzenlemesi |
| `v1.1.0` | Yeni `delamination` sınıfı eklendi |
| `v2.0.0` | Termal modalite entegrasyonu |

### 5.2 scripts/utils/versioning.py

```python
# scripts/utils/versioning.py
from pathlib import Path
import re

def get_next_version(models_dir: Path, bump: str = "patch") -> str:
    """
    Mevcut en yüksek versiyon klasörünü bulup bir sonrakini döndürür.
    bump: 'major' | 'minor' | 'patch'
    """
    versions = [
        d.name for d in models_dir.iterdir()
        if d.is_dir() and re.match(r"v\d+\.\d+\.\d+", d.name)
    ]
    if not versions:
        return "v1.0.0"
    
    latest = sorted(versions, key=lambda v: tuple(int(x) for x in v[1:].split(".")))[-1]
    major, minor, patch = (int(x) for x in latest[1:].split("."))
    
    match bump:
        case "major": return f"v{major+1}.0.0"
        case "minor": return f"v{major}.{minor+1}.0"
        case "patch": return f"v{major}.{minor}.{patch+1}"
    
    raise ValueError(f"Geçersiz bump türü: {bump}")


def promote_run_artifacts(run_dir: Path, version: str, models_dir: Path) -> Path:
    """
    YOLO'nun runs/ çıktısını models/<version>/ içine taşır.
    """
    target = models_dir / version
    target.mkdir(parents=True, exist_ok=True)
    
    for weight in ["best.pt", "last.pt"]:
        src = run_dir / "weights" / weight
        if src.exists():
            shutil.copy2(src, target / weight)
    
    # Sembolik link güncelle: models/latest -> models/<version>
    latest_link = models_dir / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(version)
    
    return target
```

---

## 6. Çift Modlu Dağıtım Stratejisi

Sistem, iki farklı donanım hedefi için optimize edilmiş bağımsız konfigürasyon profilleriyle yönetilir. Mod seçimi eğitim aşamasında belirlenir; aynı model ağırlığı her iki moda da taşınamaz — her mod için ayrı eğitim döngüsü çalıştırılmalıdır.

### 6.1 Mode 1: Edge Dağıtımı

**Hedef Donanım:** Raspberry Pi 4/5, NVIDIA Jetson Nano/Orin, Drone FPV, Mobil (Android/iOS TFLite)

**Tasarım Öncelikleri:** Düşük gecikme süresi (latency), kısıtlı VRAM/RAM, düşük güç tüketimi, ağ bağlantısına bağımsız çalışabilme.

```bash
# Edge modu eğitimi
uv run scripts/train.py \
  --config configs/train_edge.yaml \
  --model yolo11n.pt \
  --mode edge \
  --version-bump patch
```

**configs/train_edge.yaml:**

```yaml
# Edge Dağıtım Profili
model: yolo11n.pt           # Alternatif: yolo11s.pt (daha iyi doğruluk, %40 daha yavaş)
data: configs/dataset.yaml
imgsz: 416                  # 640 yerine 416: edge'de %35 hız artışı
epochs: 100
batch: -1                   # AutoBatch: OOM engellemek için otomatik
workers: 2                  # Sınırlı CPU çekirdeği için
device: cpu                 # veya 'cuda:0', 'mps'
half: false                 # Jetson için true yapılabilir (FP16)
cos_lr: true
patience: 20                # Early stopping
project: runs/detect
name: edge_train

# Inference sonrası optimizasyon hedefleri
export:
  formats: [onnx, tflite]
  dynamic: false
  simplify: true
  int8: true                # Raspberry Pi için INT8 kuantizasyonu
```

**Edge için Performans Beklentileri:**

| Donanım | Model | Çözünürlük | FPS | Gecikme |
|---|---|---|---|---|
| Raspberry Pi 4 | yolo11n (INT8) | 416×416 | ~8 | ~125ms |
| Jetson Nano | yolo11n (FP16) | 416×416 | ~22 | ~45ms |
| Jetson Orin NX | yolo11s (FP16) | 640×640 | ~35 | ~28ms |

### 6.2 Mode 2: Host/Server Dağıtımı

**Hedef Donanım:** Yüksek performanslı GPU sunucusu (RTX 3090/4090, A100, V100), bulut VM (AWS EC2 g4dn, GCP T4)

**Tasarım Öncelikleri:** Maksimum doğruluk (mAP), yüksek çözünürlük işleme, toplu iş (batch) kapasitesi, RTSP stream sunuculuğu.

```bash
# Host modu eğitimi
uv run scripts/train.py \
  --config configs/train_host.yaml \
  --model yolo11m.pt \
  --mode host \
  --version-bump minor
```

**configs/train_host.yaml:**

```yaml
# Host/Server Dağıtım Profili
model: yolo11m.pt           # Alternatif: yolo11l.pt, yolo11x.pt
data: configs/dataset.yaml
imgsz: 1280                 # Yüksek çözünürlük: ince çatlak tespiti için kritik
epochs: 200
batch: -1                   # AutoBatch
workers: 8
device: "0,1"               # Çoklu GPU (DataParallel)
half: true                  # FP16: VRAM tasarrufu ve hız
cos_lr: true
patience: 30
amp: true                   # Otomatik Mixed Precision
project: runs/detect
name: host_train

# Kaydedilen checkpoint aralığı
save_period: 10             # Her 10 epoch'ta checkpoint
```

**Host Server için Performans Beklentileri:**

| Donanım | Model | Çözünürlük | FPS | mAP50-95 (beklenen) |
|---|---|---|---|---|
| RTX 3090 | yolo11m (FP16) | 1280×1280 | ~45 | ~0.72 |
| RTX 4090 | yolo11l (FP16) | 1280×1280 | ~68 | ~0.78 |
| A100 | yolo11x (FP16) | 1280×1280 | ~55 | ~0.82 |

### 6.3 Mod Seçim Karar Ağacı

```
Cihazda işlem yapılacak mı?
    │
    ├── EVET → Cihazda GPU/NPU var mı?
    │           ├── HAYIR (sadece CPU/ARM) → yolo11n + INT8 + 416px → Edge
    │           └── EVET (Jetson vb.)     → yolo11s + FP16 + 640px → Edge+
    │
    └── HAYIR → Sunucu işleyip stream mi gönderecek?
                ├── EVET, düşük gecikme → yolo11m + FP16 + 640px → Host Balanced
                └── EVET, max doğruluk  → yolo11l/x + FP16 + 1280px → Host Accurate
```

---

## 7. Gelişmiş Eğitim Boru Hattı

### 7.1 scripts/train.py — Ana Eğitim Modülü

`train.py` modülü, YOLO API'si üzerine kurumsal düzeyde bir sarmalayıcı (wrapper) işlevi görür. Doğrudan `yolo train` CLI komutunu çağırmak yerine bu modül kullanılır; çünkü bu modül versiyonlama, loglama, kurtarma ve transfer öğrenme mantığını koordine eder.

**Tam CLI Arayüzü:**

```bash
uv run scripts/train.py [SEÇENEKLER]

Seçenekler:
  --config      PATH    Eğitim YAML konfigürasyonu (zorunlu)
  --model       TEXT    Temel model ağırlıkları [yolo11n.pt|yolo11s.pt|yolo11m.pt|...]
  --mode        TEXT    Dağıtım modu [edge|host]
  --version-bump TEXT   Versiyon artırma [major|minor|patch] (varsayılan: patch)
  --resume      PATH    Yarım kalan eğitim için last.pt yolu
  --finetune    PATH    Transfer learning için temel model (models/v1.0.0/best.pt)
  --freeze      INT     Dondurulacak katman sayısı (0 = yok, -1 = backbone tamamı)
  --focal-loss          Focal Loss'u etkinleştir (azınlık sınıfı ağırlıklandırması)
  --cos-lr              Cosine Annealing LR scheduler'ı etkinleştir
  --dry-run             Boru hattını doğrula, eğitim başlatma
  --tag         TEXT    metadata.json'a ek etiket (örn: "hotspot_focused")
  --notify              Eğitim bitiminde bildirim gönder (webhook/email)
```

### 7.2 Eğitim Boru Hattı Akış Şeması

```
train.py çalıştırıldı
         │
         ▼
┌─────────────────────┐
│  Konfigürasyon      │ ← YAML + CLI argümanları birleştir
│  Doğrulama          │   Pydantic ile şema doğrulama
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Cihaz Algılama     │ ← CUDA/MPS/CPU, VRAM kontrolü
│  & AutoBatch        │   --batch -1 → optimal batch size
└─────────┬───────────┘
          │
          ├──[--resume]──────────────────────────────┐
          │                                           ▼
          ├──[--finetune]──┐             ┌────────────────────────┐
          │                ▼             │  last.pt'den yükle     │
          │   ┌────────────────────┐     │  Epoch/optimizer       │
          │   │  Temel ağırlıkları │     │  state'i geri yükle    │
          │   │  yükle + freeze    │     └────────────┬───────────┘
          │   │  (N katman)        │                  │
          │   └────────────┬───────┘                  │
          │                │                          │
          ▼                ▼                          ▼
┌──────────────────────────────────────────────────────────┐
│                    YOLO Eğitim Döngüsü                   │
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │  Her Epoch:                                      │   │
│   │  1. Forward pass + Loss hesapla                  │   │
│   │     [--focal-loss] → FL(p) = -α(1-p)^γ log(p)  │   │
│   │  2. Backward pass + Gradient clip                │   │
│   │  3. LR güncelle [--cos-lr] → Cosine Annealing   │   │
│   │  4. Val metrikleri → mAP50, mAP50-95            │   │
│   │  5. best.pt güncelle (en iyi val mAP)            │   │
│   │  6. Her save_period'da checkpoint kaydet         │   │
│   └──────────────────────────────────────────────────┘   │
└─────────────────────────────────┬────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Post-Training           │
                    │  1. runs/ → models/vX.Y.Z/
                    │  2. metadata.json yaz    │
                    │  3. models/latest güncelle│
                    │  4. [--notify] bildirim  │
                    └─────────────────────────┘
```

### 7.3 Save & Versioning (Otomatik Artifact Yönetimi)

Eğitim tamamlandığında veya `KeyboardInterrupt` ile durdurulduğunda, `train.py` aşağıdaki post-processing adımlarını otomatik çalıştırır:

```python
# scripts/train.py — post_training() metodu (sadeleştirilmiş)
def post_training(results, args, run_dir: Path) -> Path:
    version = get_next_version(Path("models"), bump=args.version_bump)
    artifact_dir = promote_run_artifacts(run_dir, version, Path("models"))
    
    meta = build_metadata(version, args, results, run_dir)
    (artifact_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
    
    # Rapor klasörünü oluştur ve grafikleri kopyala
    report_dir = Path("reports") / version
    report_dir.mkdir(parents=True, exist_ok=True)
    for plot in run_dir.glob("*.png"):
        shutil.copy2(plot, report_dir)
    shutil.copy2(run_dir / "results.csv", report_dir)
    
    logger.info("artifact_saved", version=version, path=str(artifact_dir))
    return artifact_dir
```

### 7.4 Recovery — Eğitim Kurtarma

Sistem çökmesi, güç kesintisi veya manuel durdurma (Ctrl+C) sonrasında eğitim tam kaldığı yerden devam ettirilir:

```bash
# Eğitimi en son checkpoint'ten devam ettir
uv run scripts/train.py \
  --config configs/train_host.yaml \
  --resume models/v1.0.0/last.pt

# Veya doğrudan runs/ içindeki ham checkpoint'ten
uv run scripts/train.py \
  --config configs/train_host.yaml \
  --resume runs/detect/host_train_20250115_103200/weights/last.pt
```

`--resume` kullanıldığında:
- Epoch sayacı kaldığı yerden devam eder.
- Optimizer state (momentum, Adam momentleri) tam olarak geri yüklenir.
- LR scheduler state korunur, Cosine Annealing kesintisiz devam eder.
- `metadata.json`'a `resumed_from` alanı yazılır.

### 7.5 Transfer Learning ve Fine-Tuning

#### Temel Fine-Tuning (Tüm Katmanlar Açık):

```bash
# v1.0.0 üzerine yeni sınıfla devam
uv run scripts/train.py \
  --config configs/train_host.yaml \
  --finetune models/v1.0.0/best.pt \
  --version-bump minor
```

#### Backbone Dondurma (Frozen Feature Extraction):

```bash
# İlk 10 katmanı dondur, sadece detection head'i eğit
uv run scripts/train.py \
  --config configs/train_edge.yaml \
  --finetune models/v1.0.0/best.pt \
  --freeze 10 \
  --version-bump patch \
  --tag "hotspot_headonly"

# Tüm backbone'u dondur (-1: otomatik tespit)
uv run scripts/train.py \
  --config configs/train_edge.yaml \
  --finetune models/v1.0.0/best.pt \
  --freeze -1 \
  --version-bump patch
```

**YOLOv11 Mimari Katman Referansı (Dondurma Kılavuzu):**

| Katman Aralığı | Bileşen | Dondurma Senaryosu |
|---|---|---|
| 0–9 | Backbone (C3k2 + SPPF) | Genel özellik çıkarımı sabit, yalnızca head eğitimi |
| 10–19 | Neck (C2PSA + Upsample) | Çoklu ölçek füzyon katmanları |
| 20–22 | Detection Head | Her zaman eğitilmeli |

### 7.6 Aşırı Ezberleme ve Dengesizlik Önlemleri

**Focal Loss:**

```bash
# Azınlık sınıfı (örn: hotspot) için Focal Loss etkinleştir
uv run scripts/train.py \
  --config configs/train_host.yaml \
  --focal-loss \
  --model yolo11m.pt
```

Focal Loss: $FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$

- $\gamma = 2.0$ (varsayılan): Kolay örneklerin ağırlığını azaltır.
- $\alpha$: Sınıf frekansına göre otomatik hesaplanır.

**Cosine Annealing:**

```bash
uv run scripts/train.py \
  --config configs/train_host.yaml \
  --cos-lr
```

LR(t) = $\eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t\pi}{T}))$

**AutoBatch (OOM Önleme):**

```yaml
# configs/train_host.yaml içinde
batch: -1  # YOLO, mevcut VRAM'ın %60'ına sığacak batch boyutunu otomatik bulur
```

**Erken Durdurma:**

```yaml
patience: 30  # 30 epoch boyunca val mAP iyileşmezse dur
```

---

## 8. Doğrulama ve Raporlama

### 8.1 scripts/validate.py

```bash
# Belirli bir model versiyonunu doğrula
uv run scripts/validate.py \
  --weights models/v1.0.1/best.pt \
  --data configs/dataset.yaml \
  --split test \
  --save-dir reports/v1.0.1/ \
  --conf 0.25 \
  --iou 0.5 \
  --verbose

# Tüm versiyonları karşılaştır (regression test)
uv run scripts/validate.py \
  --compare-all \
  --output reports/comparison_v100_v101.html
```

**Üretilen Çıktılar (`reports/v1.0.1/`):**

| Dosya | Açıklama |
|---|---|
| `confusion_matrix.png` | Ham sayım karışıklık matrisi |
| `confusion_matrix_normalized.png` | Normalize edilmiş (satır yüzdesi) |
| `PR_curve.png` | Tüm sınıflar için Precision-Recall eğrisi |
| `F1_curve.png` | Confidence eşiğine göre F1 skoru |
| `results.csv` | Epoch bazlı eğitim metrikleri |
| `val_summary.json` | Özet metrikler (mAP50, mAP50-95, P, R, F1) |
| `labels.jpg` | Veri seti etiket dağılım görselleştirmesi |
| `val_batch*.jpg` | Örnek doğrulama tahminleri |

**val_summary.json Yapısı:**

```json
{
  "model_version": "v1.0.1",
  "evaluated_at": "2025-01-15T14:22:00Z",
  "split": "test",
  "thresholds": { "conf": 0.25, "iou": 0.5 },
  "overall": {
    "mAP50": 0.847,
    "mAP50_95": 0.631,
    "precision": 0.873,
    "recall": 0.812,
    "f1": 0.841
  },
  "per_class": {
    "crack":   { "mAP50": 0.912, "precision": 0.921, "recall": 0.887 },
    "dirt":    { "mAP50": 0.876, "precision": 0.853, "recall": 0.834 },
    "shadow":  { "mAP50": 0.834, "precision": 0.871, "recall": 0.809 },
    "hotspot": { "mAP50": 0.766, "precision": 0.847, "recall": 0.718 }
  }
}
```

### 8.2 Model Regresyon Testi

Yeni bir model versiyonunun öncekine göre gerileme yaşayıp yaşamadığını kontrol etmek için:

```bash
uv run scripts/validate.py \
  --weights models/v1.0.1/best.pt \
  --baseline models/v1.0.0/best.pt \
  --data configs/dataset.yaml \
  --split test \
  --regression-threshold 0.02  # mAP50'de %2'den fazla düşüşe izin verme
```

---

## 9. Canlı Çıkarım (Live Inference)

### 9.1 scripts/predict_live.py

Bu modül, eğitilmiş modeli gerçek zamanlı görüntü kaynaklarına karşı çalıştırır. Hem Edge hem de Host senaryolarını destekler.

```bash
uv run scripts/predict_live.py [SEÇENEKLER]

Seçenekler:
  --weights     PATH    Model ağırlıkları [zorunlu]
  --source      TEXT    Görüntü kaynağı [zorunlu; aşağıya bak]
  --conf        FLOAT   Tespit güven eşiği (varsayılan: 0.25)
  --iou         FLOAT   NMS IoU eşiği (varsayılan: 0.45)
  --imgsz       INT     Çıkarım görüntü boyutu (varsayılan: konfigürasyondan)
  --device      TEXT    Cihaz [cpu|cuda:0|mps]
  --half                FP16 modunda çalış (GPU gerektirir)
  --save-video  PATH    Tespit sonuçlarını videoya kaydet
  --save-frames PATH    Her karede tespit edilen kareler klasörüne kaydet
  --show                Gerçek zamanlı ekran görüntüsü (GUI gerektirir)
  --stream-out  TEXT    Tespit sonuçlarını RTSP/RTMP olarak yayınla
  --log-detections      Her tespiti JSON formatında logla
  --max-det     INT     Kare başına maksimum tespit sayısı (varsayılan: 300)
```

**Desteklenen Kaynak Formatları (`--source`):**

```
Kamera        : 0, 1, 2  (cihaz indeksi)
Görüntü       : path/to/image.jpg
Görüntü klasörü: path/to/images/
Video dosyası : path/to/video.mp4
RTSP stream   : rtsp://username:password@192.168.1.100:554/stream1
HTTP stream   : http://192.168.1.100:8080/video
YouTube       : https://youtu.be/video_id  (yt-dlp gerektirir)
Wildcard      : path/to/*.jpg
```

**Kullanım Örnekleri:**

```bash
# Lokal webcam ile canlı tespit
uv run scripts/predict_live.py \
  --weights models/latest/best.pt \
  --source 0 \
  --show

# Drone RTSP stream'i işle ve kaydet
uv run scripts/predict_live.py \
  --weights models/v1.0.1/best.pt \
  --source "rtsp://drone:pass@192.168.4.1:554/live" \
  --conf 0.30 \
  --save-video output/drone_inspection_20250115.mp4 \
  --log-detections \
  --device cuda:0 \
  --half

# Edge cihazda CPU ile çalıştır
uv run scripts/predict_live.py \
  --weights models/v1.0.0/best.pt \
  --source 0 \
  --device cpu \
  --imgsz 416 \
  --conf 0.40  # Edge'de daha yüksek eşik: FP azaltma

# Toplu görüntü klasörü işleme
uv run scripts/predict_live.py \
  --weights models/latest/best.pt \
  --source data/raw_data/rgb/images/ \
  --save-frames output/predictions/ \
  --conf 0.25
```

### 9.2 Çıkarım Mimarisi

```
[Kaynak] → [FrameReader] → [Preprocess] → [Inference] → [Postprocess] → [Output]
   │              │              │               │              │              │
RTSP/Cam    Buffer Queue    Resize+Norm      YOLO model     NMS+Filter    Ekran/Video/
Video/Img   (asyncio)       (letterbox)      forward()      (conf+iou)    Log/Stream
```

**Yüksek Gecikmeli Stream Yönetimi:**

`predict_live.py` dahili bir `asyncio` tabanlı kare tamponu kullanır. Çıkarım CPU/GPU kapasitesini aşarsa, eski kareler düşürülür ve her zaman en güncel kare işlenir. Bu, gerçek zamanlı uygulamalarda görüntü gecikmesini önler.

### 9.3 scripts/export_model.py — Model Dışa Aktarma

Eğitilmiş `.pt` modelini hedef platforma uygun formata dönüştürür:

```bash
# ONNX (evrensel, CPU/GPU)
uv run scripts/export_model.py \
  --weights models/v1.0.1/best.pt \
  --format onnx \
  --dynamic \
  --simplify

# TensorRT (NVIDIA GPU, maksimum hız)
uv run scripts/export_model.py \
  --weights models/v1.0.1/best.pt \
  --format engine \
  --half \
  --device cuda:0

# TFLite INT8 (Raspberry Pi / Coral Edge TPU)
uv run scripts/export_model.py \
  --weights models/v1.0.1/best.pt \
  --format tflite \
  --int8 \
  --data configs/dataset.yaml  # Kalibrasyonverisi için

# CoreML (Apple Silicon / iOS)
uv run scripts/export_model.py \
  --weights models/v1.0.1/best.pt \
  --format coreml \
  --half
```

---

## 10. Yapılandırma Yönetimi

### 10.1 Konfigürasyon Hiyerarşisi

Konfigürasyonlar, öncelik sırasına göre katmanlı olarak birleştirilir:

```
Öncelik (düşük → yüksek):
  1. Varsayılan değerler (scripts/utils/config_loader.py içinde)
  2. configs/*.yaml dosyaları
  3. .env dosyası
  4. CLI argümanları (en yüksek öncelik)
```

### 10.2 configs/inference.yaml

```yaml
# Üretim çıkarım parametreleri
conf_threshold: 0.25        # Genel güven eşiği
iou_threshold: 0.45         # NMS IoU eşiği
max_detections: 300
agnostic_nms: false         # Sınıf-agnostik NMS (örtüşen sınıflar için true)
multi_label: false

# Sınıf bazlı eşik ayarları (isteğe bağlı)
class_thresholds:
  hotspot: 0.20             # Azınlık sınıfı: daha düşük eşik
  crack:   0.30
  dirt:    0.25
  shadow:  0.25

# Stream kayıt ayarları
recording:
  codec: "mp4v"
  fps: 30
  output_dir: "output/recordings/"
```

### 10.3 Ortam Değişkenleri (.env.example)

```bash
# Model ve Veri
SOLAR_MODELS_DIR=models
SOLAR_DATA_DIR=data
SOLAR_REPORTS_DIR=reports

# Logging
LOG_LEVEL=INFO              # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT=json             # json | text
LOG_OUTPUT=logs/app.log     # stdout ise boş bırak

# Bildirimler (opsiyonel)
WEBHOOK_URL=https://hooks.slack.com/services/...
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
NOTIFY_EMAIL=team@company.com

# Güvenlik
RTSP_USERNAME=camera_user
RTSP_PASSWORD=secure_password_here

# MLflow (opsiyonel entegrasyon)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=solar-panel-detection
```

---

## 11. CI/CD ve Kalite Kapıları

### 11.1 Pre-commit Hook'ları

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.6.0]
```

### 11.2 GitHub Actions — lint_and_test.yml

```yaml
name: Lint & Test
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1

      - name: Ortamı kur
        run: uv sync --group dev

      - name: Lint (Ruff)
        run: uv run ruff check scripts/ tests/

      - name: Type Check (mypy)
        run: uv run mypy scripts/

      - name: Birim Testleri
        run: uv run pytest tests/ -v --cov=scripts --cov-report=xml

      - name: Veri seti boru hattı doğrulama (dry-run)
        run: uv run scripts/train.py --config configs/train_edge.yaml --dry-run
```

### 11.3 GitHub Actions — model_validation.yml

```yaml
name: Model Validation
on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  validate:
    runs-on: [self-hosted, gpu]
    steps:
      - name: Model doğrula
        run: |
          uv run scripts/validate.py \
            --weights models/${{ github.ref_name }}/best.pt \
            --baseline models/latest/best.pt \
            --data configs/dataset.yaml \
            --split test \
            --regression-threshold 0.02

      - name: Artifact'ları yükle
        uses: actions/upload-artifact@v4
        with:
          name: validation-report-${{ github.ref_name }}
          path: reports/${{ github.ref_name }}/
```

---

## 12. Gözlemlenebilirlik ve Loglama

### 12.1 Yapılandırılmış JSON Loglama

Tüm modüller `scripts/utils/logger.py` üzerinden yapılandırılmış (structured) log üretir. Bu formatı, log aggregation araçları (ELK Stack, Grafana Loki) doğrudan işleyebilir.

```python
# scripts/utils/logger.py
import structlog

def get_logger(module_name: str):
    return structlog.get_logger(module=module_name)

# Kullanım örneği:
logger = get_logger("train")
logger.info("epoch_complete",
    epoch=45, total=150,
    train_loss=0.342, val_mAP50=0.821,
    lr=0.0003, time_elapsed=127.4
)
```

**Örnek Log Çıktısı:**

```json
{
  "timestamp": "2025-01-15T10:45:32.114Z",
  "level": "info",
  "module": "train",
  "event": "epoch_complete",
  "epoch": 45,
  "total": 150,
  "train_loss": 0.342,
  "val_mAP50": 0.821,
  "lr": 0.0003,
  "time_elapsed": 127.4
}
```

### 12.2 MLflow Entegrasyonu (Opsiyonel)

```bash
# MLflow tracking server başlat
uv run mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow_artifacts

# Eğitimi MLflow ile çalıştır
MLFLOW_TRACKING_URI=http://localhost:5000 \
uv run scripts/train.py \
  --config configs/train_host.yaml \
  --model yolo11m.pt
```

MLflow entegrasyonu etkinleştirildiğinde `train.py` otomatik olarak şunları kaydeder: tüm hiperparametreler, her epoch metriği, best.pt artifact'ı, confusion matrix grafikleri.

### 12.3 scripts/benchmark.py — Performans Ölçümü

```bash
# Model performansını ölç (FPS, gecikme, bellek)
uv run scripts/benchmark.py \
  --weights models/v1.0.1/best.pt \
  --imgsz 640 \
  --device cuda:0 \
  --warmup 10 \
  --runs 100 \
  --batch 1

# Çıktı:
# ┌──────────────────────────────────┐
# │ Benchmark: v1.0.1 @ cuda:0      │
# │ Image Size: 640×640              │
# │ Batch Size: 1                    │
# ├──────────────────────────────────┤
# │ Mean Latency:    18.4ms          │
# │ P95 Latency:     21.2ms          │
# │ P99 Latency:     24.8ms          │
# │ Throughput:      54.3 FPS        │
# │ GPU Memory:      2.1 GB          │
# └──────────────────────────────────┘
```

---

## 13. Güvenlik ve Erişim Kontrolü

### 13.1 Kimlik Bilgisi Yönetimi

```bash
# ❌ YANLIŞ — Asla yapma
uv run scripts/predict_live.py \
  --source "rtsp://admin:password123@192.168.1.100/stream"

# ✅ DOĞRU — Ortam değişkeni kullan
export RTSP_USERNAME=admin
export RTSP_PASSWORD=password123
uv run scripts/predict_live.py \
  --source "rtsp://${RTSP_USERNAME}:${RTSP_PASSWORD}@192.168.1.100/stream"

# veya .env dosyasından otomatik yükle (python-dotenv)
uv run scripts/predict_live.py \
  --source "rtsp://camera1/stream"  # kimlik bilgileri .env'den okunur
```

### 13.2 Dosya Bütünlüğü Doğrulaması

```bash
# Veri seti SHA-256 parmak izi doğrulama
uv run scripts/dataset_prep.py --verify --source data/processed_data/rgb/

# Model ağırlığı doğrulama
uv run scripts/validate.py --verify-weights models/v1.0.1/best.pt
```

---

## 14. Termal Veri Entegrasyonu (Gelecek)

Bu mimari, termal kamera verilerini şu anki RGB boru hattına entegre etmek için açıkça tasarlanmıştır. Termal entegrasyon, mevcut hiçbir modülün imzasını bozmadan gerçekleştirilebilir.

### 14.1 Planlanan Genişletme Adımları

**Faz 1: Bağımsız Termal Modeli**

```bash
# Termal verisi için ayrı dataset.yaml
uv run scripts/dataset_prep.py \
  --source data/raw_data/thermal/ \
  --format coco \
  --output data/processed_data/thermal/ \
  --modality thermal

# Termal modeli eğit (tek kanal → RGB dönüştürme ile)
uv run scripts/train.py \
  --config configs/train_host_thermal.yaml \
  --model yolo11m.pt \
  --modality thermal \
  --version-bump major   # v2.0.0
```

**Faz 2: Erken Füzyon (Early Fusion)**

RGB + Termal görüntülerin kanal düzeyinde birleştirilerek 4 kanallı giriş oluşturulması:

```python
# 3 kanal RGB + 1 kanal Termal = 4 kanallı giriş
# Model giriş katmanı yeniden yapılandırılır:
# Conv1(in_channels=3) → Conv1(in_channels=4)
```

**Faz 3: Geç Füzyon (Late Fusion)**

İki bağımsız modelin tahminlerini birleştiren ensemble mimarisi:

```
RGB Modeli  ──→ [Detections_RGB]  ──┐
                                     ├──→ [Ensemble NMS] ──→ Final Detections
Thermal Modeli ──→ [Detections_IR] ──┘
```

### 14.2 Geriye Dönük Uyumluluk Garantisi

```yaml
# configs/dataset.yaml — ek alan (mevcut modeller etkilenmez)
modality: rgb           # rgb | thermal | fusion

# predict_live.py
--modality rgb          # varsayılan: rgb (mevcut davranış korunur)
--modality thermal      # termal mod
--modality fusion       # erken/geç füzyon modu
```

---

## 15. Mimari Karar Kayıtları (ADR)

### ADR-001: Neden uv?

**Bağlam:** Python bağımlılık yönetimi için `pip+venv`, `conda`, `poetry`, `uv` arasında seçim.

**Karar:** `uv` seçildi.

**Gerekçe:** `uv`, `pip`'e kıyasla 10–100× daha hızlı paket kurulumu sağlar; Rust tabanlı olduğu için platform bağımlılığı minimumdur. `uv.lock` ile deterministik ortam garantisi sunulur. `pyproject.toml` standardını tam destekler.

---

### ADR-002: Neden Monolitik YOLO CLI Yerine Modüler scripts/?

**Bağlam:** `yolo train data=... model=...` CLI doğrudan kullanılabilirdi.

**Karar:** `scripts/` dizininde Python modülleri tercih edildi.

**Gerekçe:** YOLO CLI, versiyonlama, metadata yazma, MLflow entegrasyonu, özel callback'ler ve test edilebilirlik için yetersizdir. `scripts/train.py`, YOLO API'sini programatik olarak sarmalayarak tüm kurumsal özellikleri etkinleştirir.

---

### ADR-003: Neden Semver Versiyonlama?

**Bağlam:** Model versiyonlaması için timestamp tabanlı (`20250115_103200`) veya sequential (`v1`, `v2`) yöntemler de kullanılabilirdi.

**Karar:** `vMAJOR.MINOR.PATCH` SemVer şeması.

**Gerekçe:** Semver, değişikliğin büyüklüğünü (breaking/feature/fix) versiyon numarasında kodlar. Bu, CI/CD regresyon testlerinde eşik kararlarını otomatikleştirir ve modeller arasında kırılma noktalarını açıkça belirtir.

---

### ADR-004: runs/ Neden .gitignore'da?

**Bağlam:** `runs/` dizini eğitim çıktılarını (ağırlıklar, grafikler) içerir.

**Karar:** `runs/` `.gitignore`'a eklendi; yalnızca `models/` ve `reports/` versiyonlanır.

**Gerekçe:** `runs/` büyük binary dosyalar içerir (`.pt` dosyaları GB boyutundadır). Git LFS maliyet ve karmaşıklık getirir. Yalnızca `promote_run_artifacts()` tarafından seçilen artifact'lar `models/` içine kopyalanır ve Git'e commit edilir.

---

*Bu belge projeyle birlikte gelişir. Mimari kararlar değiştiğinde önce bu dosya güncellenir, ardından kod değişiklikleri yapılır.*

---

**Son Güncelleme:** 2025-01-15 | **Bakımcı:** Solar Panel Detection Ekibi
