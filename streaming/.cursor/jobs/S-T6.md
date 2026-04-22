# JOB: S-T6 — UI / Kontrol Paneli
# ============================================================
# BAĞIMLILIK: S-T1, S-T2, S-T3, S-T4, S-T5 → TÜMÜ ✅
# NOT: Bu JOB en sonda yapılır
# ============================================================

## AGENT PROMPT

> Sen bir Kıdemli Python GUI Mühendisisin. Tüm core modüller (S-T1..S-T5) tamamlanmış sayılır. `ARCH_SUMMARY.md`'deki veri akışını referans al. GUI ana iş mantığına dokunmaz — yalnızca mevcut API'leri orchestrate eder.

## GÖREV TANIMI

`streaming/src/ui/control_panel.py` — Tkinter tabanlı kontrol paneli.

Bu panel ayrı bir pencere değil, OpenCV görüntüsünün yanına veya üstüne bindirilen bir kenar çubuğu olabilir. Tercih: OpenCV penceresi + Tkinter üst araç çubuğu kombinasyonu.

## INPUTS

```
READS:
  - streaming/.cursor/context/ARCH_SUMMARY.md
  - streaming/.cursor/context/CONTRACTS.md
  - streaming/src/core/*.py  (tüm core modüller)
```

## OUTPUTS

```
WRITES:
  - streaming/src/ui/control_panel.py
  - streaming/src/ui/__init__.py
  - streaming/tests/unit/test_control_panel.py  (mock ile)
```

## UI TASARIMI

### Pencere Düzeni

```
┌──────────────────────────────────────────────────┐
│  Solar Panel OD — v1.0  │  FPS: 28.3  │  ● REC  │
├──────────────────────────────────────────────────┤
│                                                  │
│          [Video Görüntüsü / Inference]           │
│                                                  │
├──────────────────────────────────────────────────┤
│ [Kaynak: Kamera ▼] [Model: Seç... ▼]            │
│ [▶ Başlat] [■ Durdur] [⏺ Kaydet] [❌ Çıkış]    │
│ Tespit: physical_damage(2), dust_particle(1)     │
└──────────────────────────────────────────────────┘
```

### Kontroller

| Element | İşlev |
|---|---|
| Kaynak Seçici (combobox) | Kamera 0, Kamera 1, Dosya Seç..., RTSP URL gir |
| Model Seçici (combobox) | `../models/` altındaki `.pt` dosyaları listelenir |
| Başlat / Durdur | Pipeline başlatır/durdurur |
| Kaydı Başlat/Durdur | `VideoRecorder.start()` / `stop()` |
| Conf Threshold (slider) | 0.0 – 1.0, canlı güncelleme |
| Çıkış | Temiz kapatma (cap.release(), writer.release()) |

### Threading Modeli

```
Ana Thread: Tkinter event loop
Worker Thread: Video okuma + inference + display
              → tkinter.after() ile UI güncelleme (thread-safe)
```

```python
import threading
self._running = threading.Event()
self._thread = threading.Thread(target=self._pipeline_loop, daemon=True)
```

### Model Dosyaları Listeleme

```python
def _list_available_models(self) -> list[Path]:
    models_dir = Path(__file__).parent.parent.parent.parent / "models"
    return sorted(models_dir.rglob("best.pt"))
```

## SELF_TEST (Görsel)

```bash
cd streaming
python main.py --dry-run   # Pencere açılır, model yüklenmez

# Gerçek test:
python main.py --source 0 --model ../models/latest/best.pt
```

## TEST GEREKSİNİMLERİ (mock ile)

- `test_model_list_populated`: `.pt` dosyaları combobox'ta görünür
- `test_start_stop_toggles_state`: başlat/durdur state geçişi doğru
- `test_recording_toggle`: kaydı başlat/durdur → Recorder çağrılır
- `test_clean_exit`: `on_close()` → tüm kaynaklar serbest bırakılır

## TAMAMLAMA PROTOKOLÜ

1. `python main.py --dry-run` → hatasız pencere açılıp kapanır
2. `python -m pytest tests/unit/test_control_panel.py` → PASS
3. `PROJECT_STATE.md` → S-T6: ✅ DONE
4. `git commit -m "S-T6: ControlPanel implemented ✅"`
