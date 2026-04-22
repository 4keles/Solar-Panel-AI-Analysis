# Solar Panel OD — Streaming
# Eğitilmiş YOLO modellerini kullanarak gerçek zamanlı solar panel hata tespiti.

## Özellikler

- 📷 **Kaynak Esnekliği**: Webcam, video dosyası veya RTSP akışı
- 🤖 **Model Seçimi**: Herhangi bir eğitilmiş `.pt` modelini yükle/değiştir
- 🎨 **Canlı Görselleştirme**: Bounding box + etiket + güven skoru
- ⏺ **Otomatik Kayıt**: İşaretlemelerle birlikte `.mp4` çıktı
- 🖥 **HUD**: FPS, kayıt göstergesi, tespit özeti

## Hızlı Başlangıç

```bash
cd streaming

# Webcam ile başlat:
python main.py --source 0 --model ../models/latest/best.pt

# Video dosyası + kayıt:
python main.py --source footage.mp4 --record

# Headless (GUI yok, sadece kayıt):
python main.py --source 0 --no-gui --record
```

## Mimari

Detaylı bilgi: `.cursor/context/ARCH_SUMMARY.md`

```
VideoSource → FrameProcessor → Annotator → [Display / Recorder]
                  ↑
             ModelLoader
```

## JOB'lar (Agent Görevleri)

| Görev | Dosya | Durum |
|---|---|---|
| S-T1: Model Loader | `.cursor/jobs/S-T1.md` | ⬜ |
| S-T2: Frame Processor | `.cursor/jobs/S-T2.md` | 🔒 |
| S-T3: Video Source | `.cursor/jobs/S-T3.md` | ⬜ |
| S-T4: Annotator | `.cursor/jobs/S-T4.md` | 🔒 |
| S-T5: Recorder | `.cursor/jobs/S-T5.md` | 🔒 |
| S-T6: UI Panel | `.cursor/jobs/S-T6.md` | 🔒 |
| S-T7: Config/CLI | `.cursor/jobs/S-T7.md` | ⬜ |

## Geliştirme

Bkz: `.cursor/context/HOW_TO_DEVELOP.md`

```bash
python -m pytest tests/ -v
mypy --strict src/
```
