# Solar Panel AI Analysis Framework

Bu proje, Güneş Panelleri üzerindeki fiziki ve elektriksel hataları/hasarları (ör. kırıklar, toz, kuş pisliği, elektriksel hotspot, diyot arızası vb.) otonom olarak saptamaya yarayan tam kapsamlı, endüstriyel standartlarda bir **Yapay Zeka (YOLO) Tespit ve İzleme Çerçevesidir (Framework)**.

Proje basit bir model veya arayüzden ibaret değildir; veri hazırlığından canlı akışa kadar uzanan **birbirinden bağımsız ama entegre çalışabilen modüllerden** oluşur. İhtiyacınıza göre yalnızca canlı akış arayüzünü (Kullanıcı Modu) kullanabilir veya kendi verilerinizle modeli geliştirmek için etiketleme ve eğitim boru hatlarını (Geliştirici Modu) çalıştırabilirsiniz.

## 📂 Proje Yapısı

```text
solar_panel_od/
├── configs/          # Eğitim ve arayüz konfigürasyon (YAML) dosyaları
├── data/             # Ham ve işlenmiş veri setleri, etiketler
├── models/           # Eğitilmiş YOLO ağırlıkları (.pt) ve TensorRT motorları (.engine)
├── scripts/          # Veri hazırlama, otonom etiketleme (Active Learning) ve eğitim betikleri
├── streaming/        # PyQt6 tabanlı canlı akış ve kontrol paneli arayüzü
├── tests/            # Pytest test dosyaları
├── .env              # Çevresel değişkenler ve konfigürasyonlar
├── pyproject.toml    # Bağımlılıklar (uv)
└── README.md
```

---

## 🚀 Başlangıç ve Kurulum

Proje Python tabanlıdır ve hız/modülerlik için `uv` paket yöneticisi kullanılarak yapılandırılmıştır.

```bash
# Repoyu klonlayın ve kök dizine girin
cd solar_panel_od

# Sanal ortamı aktif edin (uv otomatik oluşturduysa)
source .venv/bin/activate

# Tüm bağımlılıkları senkronize edin/kurun
uv sync
```

*(Gerekli bağımlılıklar: `ultralytics`, `opencv-python`, `PyQt6`, `albumentations`, `label-studio`, `structlog` vb. `.venv` içerisine kurulacaktır.)*

**Çevresel Değişkenler (`.env`):**
Proje kök dizininde bir `.env` dosyası bulunur. Eğer mevcut değilse kendiniz oluşturarak (veya var olanı düzenleyerek) gerekli yolları ve yapılandırmaları buraya girebilirsiniz.

---

## 🖥️ KULLANICI KILAVUZU: Canlı Akış & Kontrol Paneli (Streaming)

Eğitilen modelin gerçek dünyada (Drone, Telefon IP Kamerası, RTSP veya MP4) en yüksek performansla asenkron çalışmasını sağlayan modern **PyQt6 arayüzüdür.** Sadece panelleri izlemek ve analiz yapmak isteyen son kullanıcılar bu modülü kullanır.

### Arayüzü Başlatma
```bash
cd streaming
python main.py
```

### 🌟 Panelin Endüstriyel Özellikleri:
*   **Dinamik Qt6 Arayüz:** Sol Sidebar ve alt İstatistik Paneli dikey/yatay olarak esnetilebilir. Alt paneli genişlettiğinizde yazılar dinamik olarak büyür (dashboard tarzı).
*   **Çoklu Kaynak Desteği (Multi-Source):** Local Video (.mp4), Webcam/USB Kameralar (0, 1), IP Kamera / Telefon uygulamaları ve Drone RTSP/RTMP yayınları. Ağ kopmalarına karşı otomatik "Re-connect" yeteneği vardır.
*   **Stabilize Nesne İzleme (ByteTrack):** YOLO tespitlerindeki anlık titremeleri (flickering) önlemek ve nesneleri kareler arası kesintisiz takip etmek için **ByteTrack** izleme algoritması entegre edilmiştir.
*   **Gelişmiş Termal Mod:** `Thermal` mod aktifleştirildiğinde, görüntü üzerine Grayscale veya Inferno renk paletleri canlı olarak uygulanır. Ayrıca arayüzdeki **Contrast ve Brightness** ayarları ile termal görüntülerin dinamik aralığı canlı olarak optimize edilebilir.
*   **Tertemiz Kayıt & Görsel Yakalama:** Kayıtlar ve ekran görüntüleri üzerine FPS bandı vb. OSD verileri çizilmez; ham tespitler `output/` dizinine temiz olarak kaydedilir.
*   **Modüler Dosya Yönetimi:** Video ve Snapshot kayıt yolları uygulama içerisindeki "Ayarlar" bölümünden anlık seçilebilir.

---

## 🔬 GELİŞTİRİCİ KILAVUZU: Model Eğitimi & Veri Hazırlığı

Sistemin "Yapay Zeka" çekirdeğini geliştirmek, yeni veriler eklemek veya otonom etiketleme yapmak isteyen araştırmacılar/geliştiriciler içindir. Tüm işlemler `scripts/` dizinindeki betiklerle yönetilir.

### 1. İnsan-Döngülü Otonom Etiketleme (Active Learning)
Sistemde Label Studio entegrasyonu mevcuttur. Drone'dan alınan ham videoları ön-model ile işleyip otomatik etiketler üretebilir ve bunları Label Studio üzerinden manuel doğrulayabilirsiniz. *(Label Studio projenin ana bağımlılıkları arasındadır, geliştirici süreçleri için aktif olarak kullanılır).*

```bash
# Label Studio'yu başlatmak için ayrı bir terminalde:
label-studio start

# Otonom etiketleme boru hattını çalıştırmak için:
python scripts/active_learning_pipeline.py --image-dir data/raw_data/unlabeled --model models/v1.0.2/best.pt
```

### 2. Veri Zenginleştirme (Data Augmentation)
Az sayıdaki objeleri çoğaltmak (Noise, Rotate, Flip vb.) için Albumentations altyapısıyla yazılmış ve Bounding Box'ları oranlı olarak revize eden özel betik:
```bash
python scripts/augment.py --source data/processed_data/rgb_master/train --target-count 5000
```

### 3. Termal Etiket Filtreleme ve Sınıf Birleştirme
Düzensiz sınıfları `scripts/unify_classes.py` ile birleştirebilir veya sadece termalde anlamlı olan etiketleri süzebilirsiniz:
```bash
python scripts/filter_thermal_labels.py --input-labels data/labels --output-labels data/thermal_labels
```

### 4. Model Eğitimi (YOLO)
İşlenen veriler üzerinden eğitim başlatmak için:
```bash
python scripts/train.py --config scripts/schemas/train_config.yaml
# Veya direkt ultralytics üzerinden:
yolo detect train data=data/processed_data/rgb_master/data.yaml model=yolo11n.pt epochs=50 imgsz=640 batch=8
```

### 5. TensorRT (.engine) Optimizasyonu
Canlı yayınlarda (Streaming modülünde) maksimum FPS için modeli NVIDIA TensorRT formatına dönüştürebilirsiniz:
```bash
python scripts/export_engine.py --model models/v1.0.4/best.pt --imgsz 640
```

---

## 🧪 Testler
Projenin entegrasyonunu doğrulamak için `pytest` altyapısı kullanılır. Tüm testleri koşmak için:
```bash
pytest tests/
```

## 🧹 Temizlik ve Katkı
Sistemdeki `models/` dizini haricindeki ham veriler (`data/`) ve ağır `.pt`/`.log`/`.zip` kalıntıları `.gitignore` tarafından izlenmemektedir. Depoyu derlemeden önce lokaldeki devasa dosyaları silerek projeyi temiz tutabilirsiniz.

> **Solar Panel AI Analysis takımı tarafından ❤️ ile endüstri standardı kodlama prensiplerine bağlı kalınarak geliştirilmiştir.**

---

## 🔮 Future Works & To-Do (MLOps)
- [ ] **Konteynerizasyon:** Bağımlılıkların (CUDA, TensorRT) her sistemde aynı çalışması için NVIDIA Container Toolkit destekli Docker imajı.
- [ ] **Veri Versiyonlama:** Eğitim verilerindeki değişimlerin Git yerine **DVC (Data Version Control)** ile versiyonlanması.
- [ ] **CI/CD Pipeline:** Kod kalitesini korumak için GitHub Actions entegrasyonu.
- [ ] **Vision-Only Mapping (Optical Flow):** Görüntüleri uç uca ekleyip (stitching) güneş paneli matrisini çıkaracak haritalama algoritması.
