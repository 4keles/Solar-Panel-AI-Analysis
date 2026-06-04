# Solar Panel AI Analysis (Object Detection)

Bu proje, Güneş Panelleri üzerindeki fiziki ve elektriksel hataları/hasarları (ör. kırıklar, toz, kuş pisliği, elektriksel hotspot, diyot arızası vb.) otonom olarak saptamaya yarayan tam kapsamlı, endüstriyel standartlarda bir **Yapay Zeka (YOLO) Tespit ve İzleme Sistemidir**.

Proje, verinin hazırlanmasından donanım destekli (TensorRT) canlı akışa kadar uçtan uca bir boru hattı (pipeline) sunmaktadır ve iki ana fazdan oluşmaktadır:

1. **Model Eğitimi & Veri Hazırlığı (Phase 1):** Otonom etiketleme (Active Learning), zenginleştirme (Augmentation), format dönüştürücüler ve YOLO eğitimi.
2. **Canlı Akış & Kontrol Paneli (Phase 2):** Qt6 tabanlı, asenkron, çoklu video kaynağına (RTSP, IP Kamera, Yerel Video) bağlanabilen, Termal Renk Paleti entegreli profesyonel kontrol arayüzü.

---

## 🚀 Başlangıç ve Kurulum

Proje Python tabanlıdır ve modüler bir mimari kullanır. `uv` paket yöneticisi ile kurulmuştur.

```bash
# Proje kök dizininde olduğunuzdan emin olun
source .venv/bin/activate
```

*(Gerekli bağımlılıklar: `ultralytics`, `opencv-python`, `PyQt6`, `albumentations`, `structlog` vb. `.venv` içerisinde kuruludur.)*

---

## 🖥 1. Model Eğitimi & Veri Hazırlığı (Phase 1)

Yapay zeka analizlerinin kalbini oluşturan eğitim ve veri manipülasyon süreçleri `scripts/` dizinindeki betiklerle yönetilir:

### 1a. İnsan-Döngülü Otonom Etiketleme (Active Learning)
Henüz etiketlenmemiş (unlabeled) drone veya kamera görüntüleriniz mi var? Sistemi bir ön-model ile besleyip otomatik etiketler üretebilir, Label Studio üzerinden doğrulayabilirsiniz:
```bash
python scripts/active_learning_pipeline.py --image-dir data/raw_data/unlabeled --model models/v1.0.2/best.pt
```

### 1b. Veri Zenginleştirme (Data Augmentation)
Az sayıdaki objeleri çoğaltmak (Noise, Rotate, Flip vb.) için Albumentations altyapısıyla yazılmış ve YOLO etiketlerini (Bounding Box) oranlı olarak otomatik revize eden özel betik:
```bash
python scripts/augment.py --source data/processed_data/rgb_master/train --target-count 5000
```

### 1c. Termal Etiket Filtreleme ve Sınıf Birleştirme
Elinizdeki veriler çok fazla düzensiz sınıf (class) içeriyorsa `scripts/unify_classes.py` ile bunları tek bir çatı altında birleştirebilirsiniz. Ayrıca termal model eğitecekseniz, RGB görüntülerde görünen ama termalde anlamsız olan sınıfları süzmek için:
```bash
python scripts/filter_thermal_labels.py --input-labels data/labels --output-labels data/thermal_labels
```

### 1d. Model Eğitimi (YOLO Training)
İşlenen veriler üzerinden `yolo11` modeli eğitmek için:
```bash
python scripts/train.py --config scripts/schemas/train_config.yaml
# Veya direkt ultralytics üzerinden:
yolo detect train data=data/processed_data/rgb_master/data.yaml model=yolo11n.pt epochs=50 imgsz=640 batch=8
```

### 1e. TensorRT (.engine) Formatına Çevirme (Export)
Canlı yayınlarda maksimum FPS için PyTorch (`.pt`) modelini NVIDIA donanımsal optimizasyon formatı olan TensorRT (`.engine`) formatına dönüştürebilirsiniz:
```bash
python scripts/export_engine.py --model models/v1.0.4/best.pt --imgsz 640
```

---

## 📹 2. Canlı Akış & Kontrol Paneli (Phase 2 - Streaming)

Eğitilen modelin gerçek dünyada (Drone, Telefon IP Kamerası, RTSP veya MP4) en yüksek performansla asenkron çalışmasını sağlayan modern **PyQt6 arayüzüdür.** 

Main Thread (Arayüz) asla dondurulmaz; Görüntü Okuma, İşleme (YOLO) ve Kaydetme (VideoWriter) işlemleri farklı thread'lerde LIFO (Last-In-First-Out) kuyruk yapısıyla asenkron olarak yönetilir.

### Arayüzü Başlatma
```bash
cd streaming
python main.py
```

### 🌟 Panelin Endüstriyel Özellikleri:

*   **Çoklu Kaynak Desteği (Multi-Source):** 
    *   **Local Video:** Bilgisayarınızdaki `.mp4` test videoları.
    *   **Webcam / USB:** Bağlı `0, 1` numaralı lokal kameralar.
    *   **IP Camera / Telefon:** DroidCam veya IP Webcam uygulamalarından alınan (Örn: `http://192.168.1.50:8080/video`) canlı URL akışları.
    *   **Drone Akışı:** RTSP veya RTMP üzerinden drone yayınları. *Sistem ağ kopmalarına karşı otomatik "Re-connect" yeteneğine sahiptir.*
*   **Dinamik Qt6 Arayüz (VSCode Tarzı):** Sol Sidebar ve alt İstatistik Paneli (StatsPanel) dikey ve yatay `QSplitter` yapısındadır. Ekranınızı istediğiniz gibi bölebilir, alt paneli genişlettiğinizde yazıların dinamik olarak devasa boyutlara (dashboard tarzı) ulaşmasını sağlayabilirsiniz.
*   **Termal Mod & Renk Paletleri:** `Thermal` mod aktifleştirildiğinde, görüntü üzerine Grayscale veya Inferno renk paletleri canlı olarak uygulanır.
*   **Tertemiz Kayıt & Görsel Yakalama (Clean OSD):** Model yakalamaları ve video kayıtları esnasında görselin üstüne FPS bandı çizilmez, ham model tespitleri tertemiz olarak `output/` dizinine kaydedilir. Arayüzün tüm istatistik ihtiyacı zaten alttaki dinamik panelden sağlanmaktadır.
*   **Modüler Dosya Yönetimi:** Video ve Snapshot'ların nereye kaydedileceği uygulama içerisindeki "Ayarlar (Dosya Yolları Yönetimi)" bölümünden anlık seçilebilir ve `.yaml` konfigürasyonuna gömülür.

---

## 🧹 Temizlik ve Katkı (Contributing)

Sistemdeki `models/` dizini haricindeki ham veriler (`data/`) ve ağır `.pt`/`.log`/`.zip` kalıntıları `.gitignore` tarafından izlenmemektedir. 

*Not: Eğer bilgisayarınızdaki repoda eğitimlerden kalma devasa `train_thermal.log`, `yolo11n.pt` veya `models.zip` gibi dosyalar bulunuyorsa, projeyi derlemeden önce bu dosyaları yerel dizinden kaldırarak kök dizini temiz tutabilirsiniz.*

> **Solar Panel AI Analysis takımı tarafından ❤️ ile endüstri standardı kodlama prensiplerine bağlı kalınarak geliştirilmiştir.**

---

## 🔮 Future Works & To-Do (MLOps)

Sistemin V2.0 (Enterprise) sürümü için planlanan mimari güncellemeler:
- [ ] **Konteynerizasyon:** Bağımlılıkların (CUDA, TensorRT) her sistemde aynı çalışması için `NVIDIA Container Toolkit` destekli Docker İmajı oluşturulması.
- [ ] **Veri Versiyonlama:** Eğitim verilerindeki değişimlerin Git yerine **DVC (Data Version Control)** ile bulut depoları üzerinden versiyonlanması.
- [ ] **CI/CD Pipeline:** Kod kalitesini korumak için GitHub Actions ile `push` ve `pull_request` tetiklemelerinde otomatik test koşan otomasyon süreçleri.
- [ ] **Vision-Only Mapping (Optical Flow):** Sensörsüz drone kameralarından gelen görüntüleri uç uca ekleyip (stitching) güneş paneli matrisini çıkaracak görsel haritalama algoritması entegrasyonu.
