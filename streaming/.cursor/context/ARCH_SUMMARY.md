# MİMARİ ÖZET — Solar Panel Streaming Alt Projesi
# ============================================================
# Bu dosya streaming/ alt projesinin TOKEN-VERİMLİ mimari özetidir.
# Çakışma veya derin soru → CONTRACTS.md'ye git.
# Ana projeyle bağ → ../scripts/utils/ kullanılır (kopyalanmaz).
# ============================================================

## ALT PROJE AMACI

Eğitilmiş YOLO modellerini (RGB veya Termal) kullanarak:
1. **Gerçek zamanlı** kamera akışı üzerinde solar panel hata tespiti
2. **Video dosyası** üzerinde toplu / oynatmalı hata tespiti
3. Tespit edilen hataların **bounding box'lar ile birlikte kaydedilmesi**
4. Kullanıcı dostu **model seçme / yükleme arayüzü**

## KLASÖR HARİTASI

```
streaming/
├── .cursor/
│   ├── context/
│   │   ├── ARCH_SUMMARY.md   ← bu dosya
│   │   ├── CONTRACTS.md      ← modül sözleşmeleri
│   │   └── PROJECT_STATE.md  ← görev takip
│   └── jobs/
│       ├── S-T1.md  ← Model Loader
│       ├── S-T2.md  ← Frame Processor (inference engine)
│       ├── S-T3.md  ← Video Source Manager
│       ├── S-T4.md  ← Annotator / Overlay Renderer
│       ├── S-T5.md  ← Recorder (çıktı yazıcı)
│       ├── S-T6.md  ← UI / Kontrol Paneli
│       └── S-T7.md  ← Config & CLI
├── src/
│   ├── core/
│   │   ├── model_loader.py      ← S-T1: Model yükle/değiştir
│   │   ├── frame_processor.py   ← S-T2: Tek kare inference
│   │   ├── source_manager.py    ← S-T3: Kamera / Video / RTSP
│   │   ├── annotator.py         ← S-T4: bbox + label çizimi
│   │   └── recorder.py          ← S-T5: Video kayıt
│   ├── ui/
│   │   └── control_panel.py     ← S-T6: Tkinter/OpenCV GUI
│   └── utils/
│       ├── class_colors.py      ← Sınıf renk haritası
│       └── fps_counter.py       ← FPS hesaplama yardımcısı
├── configs/
│   ├── streaming.yaml           ← Ana streaming konfigürasyonu
│   └── display.yaml             ← Görsel overlay ayarları
├── tests/
│   ├── unit/
│   │   ├── test_model_loader.py
│   │   ├── test_frame_processor.py
│   │   ├── test_annotator.py
│   │   └── test_recorder.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── conftest.py
├── output/
│   └── recordings/              ← Kaydedilen videolar (gitignore)
├── main.py                      ← Entry point
├── README.md
└── pyproject.toml (veya requirements.txt)
```

## MODÜL SÖZLEŞME TABLOSU

| Modül | Alır | Verir | Yan Etki |
|---|---|---|---|
| `model_loader.py` | `.pt` yolu veya `models/latest` | `YOLO` nesnesi | Hiçbir dosyaya yazmaz |
| `source_manager.py` | source string (0, "video.mp4", "rtsp://...") | `np.ndarray` frame generator | Hiçbir dosyaya yazmaz |
| `frame_processor.py` | frame + YOLO model | `Results` nesnesi | Hiçbir dosyaya yazmaz |
| `annotator.py` | frame + `Results` | Annotated `np.ndarray` | Hiçbir dosyaya yazmaz |
| `recorder.py` | annotated frame stream | `.mp4` dosyası | `output/recordings/` altına yazar |
| `control_panel.py` | tüm modüllerin referansları | Kullanıcı event'leri | Ekrana çizer |

## VERİ AKIŞ DİYAGRAMI

```
[Kaynak: Kamera/Video/RTSP]
        ↓ frame (np.ndarray)
[source_manager.py]
        ↓ frame
[frame_processor.py] ← [model_loader.py] (YOLO model)
        ↓ Results
[annotator.py]
        ↓ annotated frame
    ┌───┴───────────┐
    ↓               ↓
[display/UI]    [recorder.py]
                    ↓
            output/recordings/*.mp4
```

## KRİTİK KISITLAMALAR

- `src/core/` modülleri birbirini doğrudan import EDEMEZ (yalnızca `main.py` veya `control_panel.py` orchestrate eder)
- Ana projenin `scripts/utils/logger.py` ve `config_loader.py` kullanılır — duplicate YASAK
- `recorder.py` dışında hiçbir modül `output/` dizinine yazamaz
- Her modülün kendi `--dry-run` / `--test` modu olmalı

## BAĞIMLILIK YÖNLERİ

```
configs/streaming.yaml  ──→  tüm src/ modülleri
../models/              ──→  model_loader.py (okur)
source_manager.py       ──→  frame_processor.py ──→  annotator.py ──→  recorder.py
model_loader.py         ──→  frame_processor.py
```

## TEKNOLOJİ YIĞINI

| Katman | Kütüphane |
|---|---|
| Inference | `ultralytics` (YOLO) |
| Video I/O | `opencv-python` (cv2) |
| GUI | `tkinter` + `Pillow` (PIL) |
| Config | `PyYAML` + proje config_loader |
| Logging | `structlog` (proje logger) |
| Test | `pytest` + `pytest-mock` |
| Type Check | `mypy --strict` |
