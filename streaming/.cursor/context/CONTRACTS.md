# MODÜL SÖZLEŞMELERİ — Streaming Alt Projesi
# ============================================================
# JOB'lar arası çakışmayı önlemek için kesin sınırlar.
# Çakışma durumunda BURAYA bak ve DUR.
# ============================================================

## SÖZLEŞME S-001 — ModelLoader Public API
**Sahip JOB:** S-T1
**Tüketiciler:** frame_processor.py, control_panel.py

```python
# src/core/model_loader.py — Kesin public API
class ModelLoader:
    def load(self, model_path: Path) -> YOLO: ...
    def load_latest(self, models_dir: Path, modality: str) -> YOLO: ...
    def get_class_names(self) -> dict[int, str]: ...
    def get_metadata(self) -> dict: ...

# model_path: mutlak yol veya ../models/latest/best.pt
# Dönen: ultralytics.YOLO nesnesi
# Hata: ModelLoadError (kendi exception'ı)
```

**Kural:** YOLO nesnesi yalnızca bu sınıf üzerinden yüklenir. `YOLO()` doğrudan çağrılamaz.

---

## SÖZLEŞME S-002 — SourceManager Public API
**Sahip JOB:** S-T3
**Tüketiciler:** main.py, control_panel.py

```python
# src/core/source_manager.py — Kesin public API
class VideoSource:
    def __init__(self, source: str | int | Path): ...
    def open(self) -> None: ...
    def read(self) -> tuple[bool, np.ndarray]: ...
    def release(self) -> None: ...
    def get_fps(self) -> float: ...
    def get_resolution(self) -> tuple[int, int]: ...
    def is_file(self) -> bool: ...
    def is_live(self) -> bool: ...

# source: 0 (webcam), "video.mp4", "rtsp://..."
# read() → (ret: bool, frame: np.ndarray HxWxC BGR)
```

**Kural:** OpenCV `VideoCapture` yalnızca bu sınıf içinde çağrılır.

---

## SÖZLEŞME S-003 — FrameProcessor Public API
**Sahip JOB:** S-T2
**Tüketiciler:** main.py, control_panel.py

```python
# src/core/frame_processor.py — Kesin public API
class FrameProcessor:
    def __init__(self, model: YOLO, conf: float, iou: float, device: str): ...
    def process(self, frame: np.ndarray) -> ProcessResult: ...

@dataclass
class ProcessResult:
    frame: np.ndarray          # Orijinal kare (değiştirilmemiş)
    detections: list[Detection]
    inference_ms: float

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # x1,y1,x2,y2 (piksel)
```

**Kural:** `ProcessResult.frame` hiçbir zaman değiştirilmez. Çizim `annotator.py`'nin sorumluluğundadır.

---

## SÖZLEŞME S-004 — Annotator Public API
**Sahip JOB:** S-T4
**Tüketiciler:** main.py, recorder.py, control_panel.py

```python
# src/core/annotator.py — Kesin public API
class Annotator:
    def __init__(self, class_colors: dict[int, tuple], config: DisplayConfig): ...
    def draw(self, frame: np.ndarray, result: ProcessResult) -> np.ndarray: ...
    def draw_hud(self, frame: np.ndarray, fps: float, source_label: str) -> np.ndarray: ...

# draw() → kopyalanmış ve üzerine çizilmiş frame döner
# Orijinal frame ASLA değiştirilmez (copy oluşturur)
```

**Kural:** `cv2.rectangle`, `cv2.putText` vb. yalnızca bu sınıf içinde çağrılır.

---

## SÖZLEŞME S-005 — Recorder Public API
**Sahip JOB:** S-T5
**Tüketiciler:** main.py, control_panel.py

```python
# src/core/recorder.py — Kesin public API
class VideoRecorder:
    def __init__(self, output_dir: Path, fps: float, resolution: tuple[int,int]): ...
    def start(self, filename: str | None = None) -> Path: ...  # Döner: çıktı dosya yolu
    def write(self, frame: np.ndarray) -> None: ...
    def stop(self) -> RecordingSummary: ...
    def is_recording(self) -> bool: ...

@dataclass
class RecordingSummary:
    output_path: Path
    frame_count: int
    duration_sec: float
    file_size_bytes: int
```

**Kural:** `output/recordings/` dışına yazılamaz. Sadece bu sınıf `VideoWriter` kullanır.

---

## SÖZLEŞME S-006 — Config Şeması
**Sahip JOB:** S-T7
**Tüketiciler:** Tüm modüller

```yaml
# configs/streaming.yaml — Kesin şema
model:
  path: ""                    # ../models/latest/best.pt veya mutlak yol
  modality: "rgb"             # "rgb" | "thermal"
  conf: 0.25
  iou: 0.45
  device: "0"                 # "cpu" | "0" | "cuda:0"

source:
  type: "camera"              # "camera" | "video" | "rtsp"
  camera_id: 0
  video_path: ""
  rtsp_url: ""

recording:
  enabled: true
  output_dir: "output/recordings"
  codec: "mp4v"
  fps_override: null          # null = kaynaktan al

display:
  show_window: true
  window_title: "Solar Panel OD"
  scale: 1.0
  show_fps: true
  show_conf: true
  show_class_label: true
```

---

## ÇAKIŞMA MATRİSİ

| Çakışan Durum | Karar |
|---|---|
| İki JOB aynı cv2 çağrısını yapıyor | CONTRACTS'a ekle, tek sahip belirle |
| Config şeması genişletme | S-007.md ile yeni JOB aç, mevcut BLOCKED |
| Ana proje utils çakışırsa | Ana projenin versiyonu esas alınır |
| Test fixture paylaşımı | `conftest.py` S-T1 tarafından yönetilir |
