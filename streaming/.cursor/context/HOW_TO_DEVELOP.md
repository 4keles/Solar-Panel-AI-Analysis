# Solar Panel OD — Streaming Alt Projesi
# ============================================================
# Hata Yönetimi, Test Stratejisi & Geliştirme Rehberi
# ============================================================

## HATA YÖNETİMİ MİMARİSİ

### Exception Hiyerarşisi

```python
# streaming/src/core/exceptions.py (S-T1'de oluşturulur)

class StreamingError(Exception):
    """Tüm streaming modülü hataları için taban sınıf."""

# Model Hataları
class ModelLoadError(StreamingError): ...
class ModelNotLoadedError(StreamingError): ...

# Kaynak Hataları
class SourceOpenError(StreamingError): ...
class SourceNotOpenError(StreamingError): ...

# İşleyici Hataları
class ProcessorNotInitializedError(StreamingError): ...

# Kayıt Hataları
class RecorderAlreadyRunningError(StreamingError): ...
class RecorderNotRunningError(StreamingError): ...
class RecorderSetupError(StreamingError): ...
```

### Hata Sınıflandırması

| Hata Türü | Strateji | Kullanıcıya Gösterme |
|---|---|---|
| Model dosyası bulunamadı | Log + UI mesajı, pipeline durur | "Model dosyası bulunamadı: {path}" |
| Kamera açılamadı | 3 kez yeniden dene, sonra dur | "Kamera {id} açılamıyor. Bağlı mı?" |
| Video dosyası bozuk | Hatalı kareyi atla, log'la | "Hatalı kare {n} atlandı" |
| RTSP bağlantı kesildi | 5 sn bekle, yeniden bağlan | "Yeniden bağlanıyor... ({n}/3)" |
| Disk dolu | Kaydı durdur, uyar | "Disk dolu! Kayıt durduruldu." |
| GPU belleği yetersiz | CPU'ya fallback, uyar | "GPU yetersiz, CPU'ya geçildi" |

### Yeniden Deneme Dekoratörü

```python
# src/utils/retry.py
from functools import wraps
import time

def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

# Kullanım (source_manager.py'de):
@retry(max_attempts=3, delay=2.0, exceptions=(SourceOpenError,))
def open(self) -> None: ...
```

---

## TEST STRATEJİSİ

### Test Piramidi

```
        [E2E / Görsel]  ← main.py --dry-run, manuel
       [Entegrasyon]    ← tests/integration/test_pipeline.py
      [Birim Testler]   ← tests/unit/test_*.py (CI'da)
```

### Birim Test Kuralları

```python
# tests/conftest.py

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

@pytest.fixture
def dummy_frame() -> np.ndarray:
    """640x480 siyah BGR kare."""
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def dummy_frame_hd() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)

@pytest.fixture
def mock_yolo_model():
    """Gerçek model yüklememek için mock YOLO."""
    model = MagicMock()
    model.names = {0: "physical_damage", 1: "bird_drop", ...}
    # predict() → mock Results nesnesi döner
    mock_result = MagicMock()
    mock_result.boxes.xyxy = []
    mock_result.boxes.conf = []
    mock_result.boxes.cls = []
    model.predict.return_value = [mock_result]
    return model

@pytest.fixture
def sample_pt_file(tmp_path: Path) -> Path:
    """Sahte .pt dosyası (ModelLoader dosya varlık testleri için)."""
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"fake_model_weights")
    return pt

@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Sahte 10 karelik test videosu."""
    import cv2
    path = tmp_path / "test.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for _ in range(10):
        out.write(np.zeros((480, 640, 3), dtype=np.uint8))
    out.release()
    return path
```

### Entegrasyon Testi

```python
# tests/integration/test_pipeline.py

def test_full_pipeline_video_to_recording(tmp_path, sample_video):
    """
    Tam pipeline: video → inference → annotate → kayıt.
    Gerçek model kullanmaz (mock), gerçek dosya oluşturur.
    """
    from src.core.model_loader import ModelLoader
    from src.core.source_manager import VideoSource
    from src.core.frame_processor import FrameProcessor
    from src.core.annotator import Annotator
    from src.core.recorder import VideoRecorder

    with patch("src.core.model_loader.YOLO") as MockYOLO:
        MockYOLO.return_value = mock_yolo_model()

        loader = ModelLoader()
        model = loader.load(Path("../yolo11n.pt"))  # mock
        source = VideoSource(str(sample_video))
        proc = FrameProcessor(model, conf=0.25, iou=0.45, device="cpu")
        ann = Annotator(class_colors={}, conf_threshold=0.0)
        rec = VideoRecorder(output_dir=tmp_path, fps=30, resolution=(640, 480))

        source.open()
        rec.start("test_integration")
        processed = 0
        while True:
            ret, frame = source.read()
            if not ret:
                break
            result = proc.process(frame)
            annotated = ann.draw(frame, result)
            rec.write(annotated)
            processed += 1
        source.release()
        summary = rec.stop()

        assert summary.frame_count == processed
        assert summary.output_path.exists()
        assert summary.file_size_bytes > 0
```

### CI/CD Komutu

```bash
# Tüm birim testler (CI'da çalışır):
python -m pytest tests/unit/ -v --tb=short

# Entegrasyon (CI'da çalışır, kamera mock'lu):
python -m pytest tests/integration/ -v --tb=short

# Tip kontrolü:
mypy --strict src/

# Stil:
ruff check src/ tests/
```

---

## GELİŞTİRME ADIMLARI (Doğrusal Yol)

### Adım 1 — Temel Kurulum (S-T7 önce)
```bash
cd streaming
# Bağımlılıklar (ana proje .venv'i kullanılabilir veya ayrı venv):
pip install opencv-python ultralytics pillow structlog pyyaml pytest pytest-mock mypy ruff
```

### Adım 2 — Paralel Başlangıç
```
S-T1 (Model Loader)     → test et → ✅
S-T3 (Source Manager)   → test et → ✅
S-T7 (Config/CLI)       → test et → ✅
```

### Adım 3 — Sıralı Core
```
S-T2 (Proc) → S-T4 (Annotator) → S-T5 (Recorder)
```

### Adım 4 — Entegrasyon Testi
```bash
python -m pytest tests/integration/ -v
```

### Adım 5 — UI (S-T6)
```bash
python main.py --dry-run        # GUI açılır/kapanır
python main.py --source 0       # Webcam testi
python main.py --source test.mp4 --record  # Video + kayıt
```

### Adım 6 — Gerçek Model ile Smoke Test
```bash
python main.py \
  --source 0 \
  --model ../models/latest/best.pt \
  --record \
  --conf 0.30 \
  --device 0     # GPU
```

---

## GELİŞTİRİCİ NOTLARI

### Cursor Agent Kullanımı

Her JOB için:
1. `.cursor/jobs/S-Tx.md` dosyasını aç
2. Agent penceresinde: *"Bu JOB dosyasındaki görevi tamamla"*
3. Agent okur → yazar → self-test çalıştırır → commit atar

### Hızlı Debug

```python
# Sadece inference test et (kaynak/UI yok):
python -c "
from pathlib import Path
import numpy as np
from src.core.model_loader import ModelLoader
from src.core.frame_processor import FrameProcessor

m = ModelLoader().load(Path('../models/latest/best.pt'))
p = FrameProcessor(m, conf=0.25, iou=0.45, device='0')
frame = np.zeros((640,640,3), dtype=np.uint8)
r = p.process(frame)
print(r.detections, r.inference_ms)
"

# Sadece kayıt test et:
python -c "
from pathlib import Path
import numpy as np
from src.core.recorder import VideoRecorder

r = VideoRecorder(Path('output/recordings'), 30.0, (640,480))
r.start()
[r.write(np.zeros((480,640,3), dtype='uint8')) for _ in range(30)]
s = r.stop()
print(s)
"
```

### GPU Olmadan Geliştirme

```yaml
# configs/streaming.yaml → test için:
model:
  device: "cpu"
  conf: 0.10    # Düşük threshold → daha fazla test tespiti
```
