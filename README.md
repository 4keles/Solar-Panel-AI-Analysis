# Solar Panel AI Analysis (Object Detection)

Bu proje, Güneş Panelleri üzerindeki fiziki ve elektriksel hataları/hasarları (ör. kırıklar, toz, kuş pisliği, elektriksel hotspot vb.) otomatik olarak saptamaya yarayan tam kapsamlı bir Yapay Zeka (YOLO) tespit sistemidir. 

Proje iki ana modülden (fazdan) oluşmaktadır:
1. **Model Eğitimi (Training & E2E Veri İşleme)**
2. **Canlı Akış (Streaming & Uygulama Arayüzü)**

---

## 🚀 Başlangıç ve Kurulum

Öncelikle projenin Python sanal ortamını (virtual environment) aktif ettiğinizden emin olun:
```bash
# Proje kök dizininde olduğunuzdan emin olun
source .venv/bin/activate
```
*(Bağımlılıklar `.venv` içerisinde kurulu durumdadır: `ultralytics`, `opencv-python`, `tkinter`, `albumentations` vd.)*

---

## 🖥 1. Model Eğitimi & Veri İşleme (Phase 1)

Sistemin kalbini oluşturan Yapay Zeka analiz kısımları bu adımda işlenir. Model beslemesi, çoğaltma, hata denetimleri ve eğitim otomasyonu için geliştirilmiş komutlar şunlardır:

### 1a. Veri Kümesini Çoğaltma (Data Augmentation)
Veri eğitim setinde bulunan az sayıdaki objeleri çoğaltmak (Noise, Rotate, Flip vb.) için Albumentations altyapısıyla yazılmış ve YOLO etiketlerini (Bounding Box) otomatik revize eden özel betik kullanılır:

```bash
python3 scripts/augment.py \
  --source data/processed_data/rgb_master/train \
  --output data/processed_data/rgb_master/train \
  --target-count 5000
```
- **Not:** Kullanılacak Flip orantıları, Rotasyon limitleri (açı) ve Gürültü (Noise) seviyelerini `configs/augmentation_pipeline.yaml` dosyasından manuel ayarlayabilirsiniz. Script aynı görselleri tekrar çoğaltmaz.

### 1b. Model Eğitimi (YOLO Training)
İşlenen verilere göre modeli baştan sona (`rgb_master` vb) fine-tune etmek ve yolo11 üzerinden eğitmek için komut satırını veya Ultralytics entegrasyonlarını çalıştırabilirsiniz:
```bash
yolo detect train data=data/processed_data/rgb_master/data.yaml model=yolo11n.pt epochs=50 imgsz=640 batch=8
```
*(Model tamamlandığında ağırlıklar (weights) `runs/detect/` klasörüne, projede kullanılan en iyi model formatı ise `models/v1.0.2/best.pt` altına taşınır.)*

### 1c. Model Doğrulama & Metrikler (Validation)
Eğitimi tamamlanmış modelin test setlerindeki kalitesini doğrulamak adına:
```bash
yolo val model=models/v1.0.2/best.pt data=data/processed_data/rgb_master/data.yaml split=val
```

---

## 📹 2. Canlı Akış / Uygulama (Phase 2 - Streaming)

Eğitilen modelin gerçek dünyada (Kamera, RTSP Bağlantısı veya MP4 Videoları üzerinden) asenkron çalışmasını sağlayan LIFO (Last-In-First-Out) kuyruk destekli Multi-Thread GUI mimarisidir. Main Thread dondurulmadan yüksek performanslı çalışabilecek şekilde tasarlanmıştır.

### Uygulamayı Başlatma (Tkinter Control Panel)
```bash
cd streaming
python3 main.py
```

### Uygulamanın Özellikleri:
- **Kaynak Seçimi:** Arayüz açıldığında kaynak (*Source*) olarak Webcam'inizi veya mp4 test videolarınızı seçebilirsiniz.
- **Dinamik Yükleme:** Parametre verdiğiniz Model (`models/v1.0.2/best.pt` vb.) otomatik taranarak asenkron belleğe yüklenir.
- **HUD & Annotator:** Akış esnasında tespit edilen sınıfların renkli Bounding Box çizimleri, doğruluk yüzdeleri ve FPS değerleri canlı olarak videonun üstüne renderlanır.
- **Video Kayıt (VideoRecorder):** Kayıt (Recording) modunu başlattığınızda işlenmiş tespit pencereleri asenkron bir Thread üzerinden `streaming/recordings/` altına `mp4v` olarak kaydedilir. Disk IO hızının analizini asla dondurmadığı, kare (Frame) atlaması yaşatmadığı onaylanmıştır.
