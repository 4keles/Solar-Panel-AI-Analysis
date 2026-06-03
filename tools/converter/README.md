# Model Dönüştürücü — Kullanım Kılavuzu

Ultralytics `.pt` modellerini daha hızlı inference formatlarına dönüştürmek için araç.

## Hız Karşılaştırması (RTX 3060, YOLOv11s, 640px)

| Format         | FPS (tahmin) | Gecikme    | Gereksinim           |
|----------------|-------------|------------|----------------------|
| `.pt` (PyTorch) | ~30–45      | ~22–33 ms  | Sadece PyTorch       |
| `.onnx`         | ~45–70      | ~14–22 ms  | onnxruntime-gpu      |
| `.engine` (TRT) | ~90–150+    | ~7–11 ms   | TensorRT ≥ 8.6       |

## GUI Başlatma

```bash
cd /home/kayra/git/solar_panel_od
python tools/converter/main.py
```

## CLI Modu

```bash
# .pt → ONNX (her sistemde çalışır)
python tools/converter/main.py --cli \
    --input models/v1.0.4/best.pt \
    --format onnx \
    --imgsz 640

# .pt → TensorRT Engine (GPU + TRT gerekli)
python tools/converter/main.py --cli \
    --input models/v1.0.4/best.pt \
    --format engine \
    --half \
    --imgsz 640

# Tüm parametreler
python tools/converter/main.py --cli --help
```

## TensorRT Kurulumu

TensorRT, NVIDIA GPU + Driver gerektirir:

```bash
# CUDA 12.x için:
pip install tensorrt==10.* --index-url https://pypi.nvidia.com

# Veya sistem TRT kullanılıyorsa:
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

> **Not**: `.engine` dosyaları oluşturulduğu GPU modeline özgüdür.
> Farklı bir GPU'ya taşırsanız yeniden dönüştürmeniz gerekir.

## Çıktı Dosyaları

Dönüştürülen modeller `models/` dizinine (orijinal modelin yanına) kaydedilir.
Streaming UI, `.engine`, `.onnx` ve `.pt` dosyalarını otomatik tanır.
