# MODÜL SÖZLEŞMELERİ (CONTRACTS)
# ============================================================
# Bu dosya paralel JOB'ların birbirinin alanına girmemesi için
# her modülün kesin sınırlarını tanımlar.
# Çakışma tespitinde BURAYA bak.
# ============================================================

## SÖZLEŞME 001 — Logger Arayüzü
**Sahip JOB:** P1-T2  
**Tüketiciler:** Tüm modüller

```python
# scripts/utils/logger.py — Kesin public API
def get_logger(name: str) -> structlog.BoundLogger: ...
# Başka hiçbir fonksiyon export edilmez.
# Kullanım: from scripts.utils.logger import get_logger
```

**Kural:** Hiçbir JOB kendi log mekanizmasını oluşturamaz.

---

## SÖZLEŞME 002 — Config Yükleyici Arayüzü
**Sahip JOB:** P1-T2  
**Tüketiciler:** train.py, validate.py, predict_live.py, dataset_prep.py

```python
# scripts/utils/config_loader.py — Kesin public API
def load_config(path: Path) -> dict: ...
def merge_configs(base: dict, override: dict) -> dict: ...
def validate_config(config: dict, schema_class: type[BaseModel]) -> BaseModel: ...
```

**Kural:** YAML okuma sadece bu modül üzerinden yapılır. `yaml.safe_load()` doğrudan çağrılamaz.

---

## SÖZLEŞME 003 — Versiyonlama Arayüzü
**Sahip JOB:** P1-T6  
**Tüketiciler:** train.py (P2-T2), post-training promoter (P2-T5)

```python
# scripts/utils/versioning.py — Kesin public API
def get_next_version(models_dir: Path, bump: BumpType) -> str: ...
def promote_run_artifacts(run_dir: Path, version: str, models_dir: Path) -> Path: ...
def update_latest_symlink(models_dir: Path, version: str) -> None: ...
def list_versions(models_dir: Path) -> list[str]: ...
```

**Kural:** `models/` dizinine doğrudan dosya yazan/kopyalayan başka JOB OLAMAZ.

---

## SÖZLEŞME 004 — Metadata Arayüzü
**Sahip JOB:** P2-T6  
**Tüketiciler:** train.py (P2-T2), validate.py (P3-T1), benchmark.py (P3-T4)

```python
# scripts/utils/metadata.py — Kesin public API
def write_metadata(path: Path, data: TrainingMetadata) -> None: ...
def read_metadata(version_dir: Path) -> TrainingMetadata: ...
def build_training_metadata(version, args, results, run_dir, git_info) -> TrainingMetadata: ...

# Pydantic model — bu sınıf dışında metadata oluşturulamaz:
class TrainingMetadata(BaseModel):
    version: str
    created_at: datetime
    base_model: str
    deployment_mode: DeployMode
    data_modality: Modality
    dataset: DatasetInfo
    hyperparameters: HyperParams
    results: TrainingResults | None
    hardware: HardwareInfo
    git: GitInfo
    resumed_from: str | None
    finetuned_from: str | None
```

---

## SÖZLEŞME 005 — Dataset Prep Çıktı Formatı
**Sahip JOB:** P1-T4  
**Tüketiciler:** train.py (P2-T2), augment.py (P1-T5), validate.py (P3-T1)

```
Kesin çıktı dizin yapısı (değiştirilemez):
data/processed_data/<modality>/
  ├── train/
  │   ├── images/   ← .jpg veya .png
  │   └── labels/   ← .txt (YOLO format: class cx cy w h, normalize)
  ├── val/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/

configs/dataset.yaml çıktı şeması:
  path: data/processed_data/<modality>
  train: train/images
  val:   val/images
  test:  test/images
  nc: <int>
  names: <dict[int, str]>
```

---

## SÖZLEŞME 006 — Train.py Çıktı Garantisi
**Sahip JOB:** P2-T2  
**Tüketiciler:** P2-T3 (resume), P2-T4 (finetune), P2-T5 (promoter)

```
runs/detect/<run_name>/
  ├── weights/
  │   ├── best.pt    ← zorunlu
  │   └── last.pt    ← zorunlu
  ├── results.csv    ← zorunlu
  ├── *.png          ← grafikler (opsiyonel)
  └── args.yaml      ← eğitim argümanları (YOLO üretir)
```

**Kural:** `train.py` dışında hiçbir JOB `runs/` dizinine yazamaz.

---

## SÖZLEŞME 007 — Validate.py Çıktı Formatı
**Sahip JOB:** P3-T1  
**Tüketiciler:** P3-T2 (reporter), P3-T3 (regression), P4-T5 (CI/CD)

```python
# validate.py'nin döndürdüğü Python dict (val_summary.json ile aynı yapı):
{
    "model_version": str,
    "evaluated_at": str,       # ISO 8601
    "split": str,              # "val" | "test"
    "thresholds": {"conf": float, "iou": float},
    "overall": {
        "mAP50": float,
        "mAP50_95": float,
        "precision": float,
        "recall": float,
        "f1": float
    },
    "per_class": {
        "<class_name>": {"mAP50": float, "precision": float, "recall": float}
    }
}
```

---

## ÇAKIŞMA KARARI MATRİSİ

| Çakışan Durum | Öncelik | Karar |
|---|---|---|
| İki JOB aynı utils fonksiyonunu yazıyor | Sahip JOB önce tamamlanır | Diğeri BLOCKED |
| Test fixture'ları çakışıyor | `conftest.py` ortak tutulan | P1-T1 yönetir |
| Config şeması genişletmek gerekiyor | Önce CONTRACTS.md güncellenir | Sonra JOB devam eder |
| `pyproject.toml`'a yeni bağımlılık | Yeni `INFRA` JOB açılır | Mevcut JOB'u blokla |
