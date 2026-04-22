# JOB: S-T3 — Video Source Manager
# ============================================================
# BAĞIMLILIK: Yok (S-T1 ile paralel)
# PARALEL: S-T1, S-T7 ile birlikte çalışabilir
# ============================================================

## AGENT PROMPT

> Sen bir Kıdemli Video İşleme Mühendisisin. `CONTRACTS.md` içindeki `SÖZLEŞME S-002`'yi oku. Bu modül SADECE video kaynağını yönetir — inference veya çizim YOK.

## GÖREV TANIMI

`streaming/src/core/source_manager.py` dosyasını yaz.

3 kaynak tipini desteklemeli:
1. **Webcam/USB Kamera** → `source = 0` (veya `1, 2, ...`)
2. **Lokal Video Dosyası** → `source = "path/to/video.mp4"`
3. **RTSP/IP Kamera** → `source = "rtsp://ip:port/path"`

## INPUTS

```
READS:
  - streaming/.cursor/context/CONTRACTS.md  (SÖZLEŞME S-002)
```

## OUTPUTS

```
WRITES:
  - streaming/src/core/source_manager.py
  - streaming/tests/unit/test_source_manager.py
```

## UYGULAMA REHBERİ

### Kaynak Tür Tespiti

```python
def _detect_source_type(self) -> SourceType:
    # int → camera, "rtsp://" başlangıcı → rtsp, Path.exists() → video file
```

### Kritik Davranışlar

- `open()` çağrılmadan `read()` → `SourceNotOpenError`
- Açılamayan kaynak (dosya yok, kamera bağlı değil) → `SourceOpenError` (detaylı mesaj ile)
- `get_fps()`: Video dosyası için CAP_PROP_FPS, canlı için genellikle 30.0 döner
- `get_resolution()` → `(width, height)` tuple
- `is_file()` → kaynak video dosyasıysa True
- `is_live()` → webcam veya RTSP ise True

### RTSP için Özel Ayarlar (buffer sıkışması önleme)

```python
if self._is_rtsp:
    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```

### Context Manager Desteği

```python
with VideoSource("video.mp4") as src:
    ret, frame = src.read()
```

### Exceptions

```python
class SourceOpenError(Exception): ...
class SourceNotOpenError(Exception): ...
```

## SELF_TEST

```bash
cd streaming
python -m pytest tests/unit/test_source_manager.py -v

# Smoke (webcam olmayan ortamda mock ile çalışır):
python -c "
from src.core.source_manager import VideoSource
# Test modu: sahte video dosyası ile
"
```

## TEST GEREKSİNİMLERİ

- `test_detect_camera_source`: `0` → SourceType.CAMERA
- `test_detect_video_source`: `"video.mp4"` → SourceType.VIDEO
- `test_detect_rtsp_source`: `"rtsp://..."` → SourceType.RTSP
- `test_read_without_open_raises`: `SourceNotOpenError`
- `test_open_missing_file_raises`: `SourceOpenError`
- `test_context_manager_closes`: `with` bloğu sonrası `_cap` serbest bırakıldı
- `test_is_file_and_is_live`: doğru bool değerleri

Mock: `cv2.VideoCapture` → `pytest-mock` ile mocklama zorunlu (gerçek kamera yok)

## TAMAMLAMA PROTOKOLÜ

1. `python -m pytest tests/unit/test_source_manager.py` → PASS
2. `mypy --strict src/core/source_manager.py`
3. `PROJECT_STATE.md` → S-T3: ✅ DONE
4. `git commit -m "S-T3: VideoSource implemented ✅"`
5. S-T3 tamamlanınca S-T6 ve S-T7 bu modülü kullanabilir
