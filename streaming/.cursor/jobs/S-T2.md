# JOB: S-T2 — Frame Processor (Inference Engine)
# ============================================================
# BAĞIMLILIK: S-T1 (ModelLoader) ✅ tamamlanmış olmalı
# PARALEL: S-T3, S-T4 paralel başlayabilir
# ============================================================

## AGENT PROMPT

> Sen bir Kıdemli ML Mühendisisin. Bu JOB'u tamamlamadan önce `CONTRACTS.md` içindeki `SÖZLEŞME S-003`'ü oku. S-T1 tamamlanmış sayılır — `ModelLoader` API'sini güvenle kullanabilirsin.

## GÖREV TANIMI

`streaming/src/core/frame_processor.py` dosyasını yaz.

Bu modül, bir video karesini alır, YOLO modeline verir ve yapılandırılmış `ProcessResult` nesnesi döner. **Görsel çizim YOK** — bu sorumluluk `annotator.py`'de (S-T4).

## INPUTS

```
READS:
  - streaming/.cursor/context/CONTRACTS.md  (SÖZLEŞME S-003)
  - streaming/src/core/model_loader.py      (S-T1 çıktısı)
```

## OUTPUTS

```
WRITES:
  - streaming/src/core/frame_processor.py
  - streaming/tests/unit/test_frame_processor.py
```

## UYGULAMA REHBERİ

### Veri Tipleri (Kesin — CONTRACTS'tan değiştirilmez)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # piksel, mutlak

@dataclass
class ProcessResult:
    frame: np.ndarray          # Orijinal kare — ASLA değiştirme
    detections: list[Detection]
    inference_ms: float        # Sadece inference süresi
```

### Performans Notları

- `model.predict()` çağrısını `time.perf_counter()` ile sar → `inference_ms`
- `conf` ve `iou` threshold'ları `__init__`'te alınır, `predict()` sırasında kullanılır
- `device` argümanı model'e değil, `predict()` çağrısına verilir (her kare için değişebilir)

### Edge Case'ler

- Boş/siyah kare gelirse: boş `detections=[]` ile döner, hata fırlatmaz
- Model henüz yüklenmemişse: `ProcessorNotInitializedError` fırlat
- YOLO `Results` nesnesi `boxes` None ise: boş `detections=[]` döner

## SELF_TEST

```bash
cd streaming
python -m pytest tests/unit/test_frame_processor.py -v

# Smoke (gerçek model ile):
python -c "
import numpy as np
from pathlib import Path
from src.core.model_loader import ModelLoader
from src.core.frame_processor import FrameProcessor
loader = ModelLoader()
model = loader.load(Path('../yolo11n.pt'))
proc = FrameProcessor(model, conf=0.25, iou=0.45, device='cpu')
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
result = proc.process(dummy)
print(f'Detections: {len(result.detections)}, ms: {result.inference_ms:.1f}')
"
```

## TEST GEREKSİNİMLERİ

- `test_process_returns_original_frame_unchanged`: frame kopyalanmaz, orijinal döner
- `test_empty_frame_returns_no_detections`: siyah kare → boş liste
- `test_detections_have_correct_fields`: Detection dataclass alanları dolu
- `test_inference_ms_positive`: süre > 0
- `test_uninitialized_raises`: model yokken `process()` → hata

## TAMAMLAMA PROTOKOLÜ

1. `python -m pytest tests/unit/test_frame_processor.py` → all PASS
2. `mypy --strict src/core/frame_processor.py`
3. `PROJECT_STATE.md` → S-T2: ✅ DONE
4. `git commit -m "S-T2: FrameProcessor implemented ✅"`
5. Sonraki JOB: **S-T4** (S-T4 S-T2'yi bekliyor)
