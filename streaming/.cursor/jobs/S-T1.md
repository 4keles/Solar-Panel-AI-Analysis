# JOB: S-T1 — Model Loader
# ============================================================
# BAĞIMLILIK: Yok
# PARALEL: S-T3, S-T7 ile birlikte çalışabilir
# ============================================================

## AGENT PROMPT

> Sen bir Kıdemli Python Mimarısın. Bu JOB'u tamamlamak için önce bu dosyayı tamamen oku, sonra `../context/CONTRACTS.md` içindeki `SÖZLEŞME S-001`'i oku. Başka dosya okuma.

## GÖREV TANIMI

`streaming/src/core/model_loader.py` dosyasını sıfırdan yaz.

Bu modül, ana projenin `models/` dizinindeki eğitilmiş YOLO modellerini yükler. Sınıf şeması `SÖZLEŞME S-001`'de tanımlanmıştır — eksiksiz uygula.

## INPUTS

```
READS:
  - streaming/.cursor/context/CONTRACTS.md  (SÖZLEŞME S-001)
  - streaming/.cursor/context/ARCH_SUMMARY.md
```

## OUTPUTS

```
WRITES:
  - streaming/src/core/model_loader.py    (ana çıktı)
  - streaming/src/core/__init__.py        (boş veya minimal export)
  - streaming/tests/unit/test_model_loader.py
```

## UYGULAMA REHBERİ

### Sınıf Davranışı
- `load(model_path)`: Verilen `.pt` dosyasını yükler. Dosya yoksa `ModelLoadError` fırlatır.
- `load_latest(models_dir, modality)`: `models_dir/latest/best.pt` veya `models_dir/latest/<modality>_best.pt` dener.
- `get_class_names()`: Son yüklenen modelin sınıf isimlerini `{0: "physical_damage", ...}` formatında döner.
- `get_metadata()`: Varsa `metadata.json`'ı okur, yoksa `{}` döner.
- Model yüklendikten sonra GPU'da mı CPU'da mı çalıştığını log'la.

### Özel Exception

```python
class ModelLoadError(Exception):
    """Raised when model file is missing or corrupted."""
```

### Logger Kullanımı

```python
# ANA PROJEDEN IMPORT — kopyalama!
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils.logger import get_logger
```

### Tip Şablonu

```python
from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass, field

class ModelLoader:
    def __init__(self) -> None:
        self._model: YOLO | None = None
        self._model_path: Path | None = None

    def load(self, model_path: Path) -> YOLO:
        """Load YOLO model from path.
        Args:
            model_path: Absolute path to .pt file.
        Returns:
            Loaded YOLO instance.
        Raises:
            ModelLoadError: If file not found or load fails.
        """
        ...
```

## SELF_TEST

```bash
# Çalıştır ve hata vermemeli:
cd streaming
python -m pytest tests/unit/test_model_loader.py -v

# Smoke test:
python -c "
from pathlib import Path
from src.core.model_loader import ModelLoader
loader = ModelLoader()
# Var olan bir .pt ile test:
# m = loader.load(Path('../yolo11n.pt'))
# print(m)
"
```

## TEST GEREKSİNİMLERİ

`tests/unit/test_model_loader.py` içinde şunlar OLMALI:
- `test_load_valid_model`: Geçerli `.pt` dosyası başarıyla yüklenir
- `test_load_missing_file_raises`: Olmayan dosya `ModelLoadError` fırlatır
- `test_get_class_names_after_load`: Yükleme sonrası sınıf isimleri doğru
- `test_load_latest_finds_best_pt`: `load_latest()` en iyi `.pt` bulur
- `test_metadata_missing_returns_empty`: `metadata.json` olmayan modelde `{}` döner

Mock: Gerçek model ağırlığı gerektiren testlerde `pytest-mock` ile `YOLO.__init__` mock'la.

## TAMAMLAMA PROTOKOLÜ

1. Test çıktısı kontrol et: tüm testler PASS
2. `mypy --strict src/core/model_loader.py` → hata yok
3. `PROJECT_STATE.md` → S-T1: ✅ DONE
4. `git add -A && git commit -m "S-T1: ModelLoader implemented ✅"`
5. Sonraki önerilen JOB: **S-T2** (S-T1 tamamlanınca BLOCKED kalkar)
