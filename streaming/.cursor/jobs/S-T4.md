# JOB: S-T4 — Annotator / Overlay Renderer
# ============================================================
# BAĞIMLILIK: S-T2 (FrameProcessor) ✅
# PARALEL: S-T5 ile paralel başlayabilir
# ============================================================

## AGENT PROMPT

> Sen bir Kıdemli Görüntü İşleme Mühendisisin. `CONTRACTS.md` → `SÖZLEŞME S-004`'ü oku. Bu modül SADECE çizim yapar — girdi framei değiştirmez, kopyasına çizer.

## GÖREV TANIMI

`streaming/src/core/annotator.py` dosyasını yaz.

`ProcessResult` nesnesini alır, her tespiti bounding box + etiket ile görselleştirir. Sonucu ekranda göstermek veya kaydetmek için hazır kare (`np.ndarray`) döner.

## INPUTS

```
READS:
  - streaming/.cursor/context/CONTRACTS.md  (SÖZLEŞME S-004)
  - streaming/src/core/frame_processor.py   (ProcessResult, Detection)
```

## OUTPUTS

```
WRITES:
  - streaming/src/core/annotator.py
  - streaming/src/utils/class_colors.py
  - streaming/tests/unit/test_annotator.py
```

## UYGULAMA REHBERİ

### Sınıf Renk Üretimi (class_colors.py)

```python
# Her sınıf için deterministik, ayırt edilebilir renk
def generate_class_colors(class_names: dict[int, str]) -> dict[int, tuple[int,int,int]]:
    """BGR tuple döner. Seed sınıf adı hashinden."""
    # Örnek: "physical_damage" → (255, 50, 50) kırmızımsı
```

Önerilen renk haritası (solar panel sınıfları için):
```python
PRESET_COLORS = {
    "physical_damage":    (0, 0, 255),      # Kırmızı
    "bird_drop":          (0, 140, 255),    # Turuncu
    "dust_particle":      (0, 255, 255),    # Sarı
    "leaf":               (0, 200, 0),     # Yeşil
    "snow":               (255, 255, 255), # Beyaz
    "healthy":            (0, 255, 0),     # Parlak yeşil
    "bird_feather":       (255, 100, 0),   # Mavi
}
```

### Çizim Davranışı

```python
def draw(self, frame: np.ndarray, result: ProcessResult) -> np.ndarray:
    annotated = frame.copy()  # Orijinal ASLA değiştirilmez
    for det in result.detections:
        # 1. Renkli bounding box (kalınlık: 2px)
        # 2. Dolgu + translucent arka plan
        # 3. "ClassName 0.87" etiketi (üst sol köşe)
    return annotated
```

### HUD (Heads-Up Display)

```python
def draw_hud(self, frame: np.ndarray, fps: float, source_label: str,
             recording: bool = False) -> np.ndarray:
    # Sol üst: FPS, Source
    # Sağ üst: Eğer kayıt varsa kırmızı ● REC
    # Alt: Tespit sayısı ve sınıf özeti
```

### FPS Counter (utils/fps_counter.py)

```python
class FPSCounter:
    def __init__(self, window: int = 30) -> None: ...  # rolling window
    def tick(self) -> None: ...
    def get_fps(self) -> float: ...
```

## SELF_TEST

```bash
cd streaming
python -m pytest tests/unit/test_annotator.py -v

# Görsel test (pencere açılır):
python -c "
import numpy as np, cv2
from src.core.annotator import Annotator
from src.core.frame_processor import ProcessResult, Detection
from src.utils.class_colors import PRESET_COLORS

ann = Annotator(class_colors=PRESET_COLORS, conf_threshold=0.25)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
result = ProcessResult(frame=frame, detections=[
    Detection(0, 'physical_damage', 0.87, (50, 50, 200, 200))
], inference_ms=12.5)
out = ann.draw(frame, result)
cv2.imshow('Test', out); cv2.waitKey(2000)
"
```

## TEST GEREKSİNİMLERİ

- `test_draw_does_not_modify_original`: `frame` aynı kalır, dönen farklıdır
- `test_draw_empty_detections`: Tespit yoksa frame kopyası döner (hatasız)
- `test_bbox_drawn_at_correct_coords`: piksel koordinatları doğru
- `test_hud_shows_fps`: HUD metni içinde "FPS" geçer
- `test_recording_indicator_shown`: `recording=True` → "REC" görünür

## TAMAMLAMA PROTOKOLÜ

1. Testler PASS
2. `mypy --strict src/core/annotator.py src/utils/class_colors.py`
3. `PROJECT_STATE.md` → S-T4: ✅ DONE
4. `git commit -m "S-T4: Annotator implemented ✅"`
5. Sonraki JOB: **S-T5** (Recorder)
