# JOB: S-T5 — Video Recorder
# ============================================================
# BAĞIMLILIK: S-T4 (Annotator) ✅
# PARALEL: S-T4 ile paralel düşünülebilir
# ============================================================

## AGENT PROMPT

> Sen bir Kıdemli Video Mühendisisin. `CONTRACTS.md` → `SÖZLEŞME S-005`'i oku. Bu modül SADECE `output/recordings/` altına yazar — başka hiçbir yere.

## GÖREV TANIMI

`streaming/src/core/recorder.py` — Annotated frame'leri `.mp4` dosyasına yazar.

## INPUTS

```
READS:
  - streaming/.cursor/context/CONTRACTS.md  (SÖZLEŞME S-005)
```

## OUTPUTS

```
WRITES:
  - streaming/src/core/recorder.py
  - streaming/tests/unit/test_recorder.py
```

## UYGULAMA REHBERİ

### Dosya Adlandırma Kuralı

```python
# Otomatik: recordings/session_YYYYMMDD_HHMMSS.mp4
# Manuel:   recordings/<kullanıcı_verdiği_isim>.mp4
def _generate_filename(self) -> str:
    from datetime import datetime
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
```

### Video Writer Başlatma

```python
fourcc = cv2.VideoWriter_fourcc(*self._codec)  # "mp4v"
self._writer = cv2.VideoWriter(
    str(output_path), fourcc, self._fps,
    (self._resolution[0], self._resolution[1])
)
```

### Thread Güvenliği

`write()` metodu ana döngüden çağrılır. `threading.Lock()` ile koruma eklenmeli:
```python
self._lock = threading.Lock()
def write(self, frame: np.ndarray) -> None:
    with self._lock:
        if self._writer and self._recording:
            self._writer.write(frame)
```

### `RecordingSummary` hesaplama

```python
@dataclass
class RecordingSummary:
    output_path: Path
    frame_count: int
    duration_sec: float   # frame_count / fps
    file_size_bytes: int  # output_path.stat().st_size
```

### Hata Durumları

- `start()` çağrılmışken tekrar `start()` → `RecorderAlreadyRunningError`
- `stop()` kayıt yokken → `RecorderNotRunningError`
- `output_dir` oluşturulamazsa → `RecorderSetupError`

## SELF_TEST

```bash
cd streaming
python -m pytest tests/unit/test_recorder.py -v

# Entegrasyon smoke:
python -c "
import numpy as np
from pathlib import Path
from src.core.recorder import VideoRecorder

rec = VideoRecorder(
    output_dir=Path('output/recordings'),
    fps=30.0,
    resolution=(640, 480)
)
out = rec.start('test_output')
for _ in range(10):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rec.write(frame)
summary = rec.stop()
print(f'Saved: {summary.output_path}, {summary.frame_count} frames')
assert summary.output_path.exists()
summary.output_path.unlink()
print('OK')
"
```

## TEST GEREKSİNİMLERİ

- `test_start_creates_file`: `start()` → dosya varlığı
- `test_write_increments_frame_count`: 5 `write()` → `frame_count=5`
- `test_stop_returns_summary`: `RecordingSummary` alanları dolu
- `test_double_start_raises`: `RecorderAlreadyRunningError`
- `test_stop_without_start_raises`: `RecorderNotRunningError`
- `test_output_only_in_recordings_dir`: çıktı `output/recordings/` altında

## TAMAMLAMA PROTOKOLÜ

1. Testler PASS + gerçek dosya oluşturuldu → silindi
2. `mypy --strict src/core/recorder.py`
3. `PROJECT_STATE.md` → S-T5: ✅ DONE
4. `git commit -m "S-T5: VideoRecorder implemented ✅"`
5. Sonraki JOB: **S-T6** (UI — tüm core tamamlanınca)
