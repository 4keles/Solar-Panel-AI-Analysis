# CURSOR PROJE DÜZENLEME PROMPTU
# ============================================================
# Bu promptu Cursor'da yeni bir sohbet başlatıp yapıştır.
# Cursor projeyi mevcut durumundan hedef mimariye taşıyacak.
# ============================================================

---

## CURSOR'A VERİLECEK PROMPT (Kopyala → Cursor Chat'e Yapıştır)

---

Sen kıdemli bir ML mühendisisin. Elimde mevcut bir güneş paneli arıza tespit projesi var. Senden bu projeyi `.cursor/` klasöründeki mimari kurallara uygun hale getirmeni istiyorum.

---

### ADIM 0 — Önce Bunları Oku (Başka Hiçbir Şeyi Okuma)

Bu dosyaları sırayla oku, sonra devam et:

1. `.cursor/context/QUICK_CONTEXT.md`
2. `.cursor/context/ARCH_SUMMARY.md`
3. `.cursor/context/CONTRACTS.md`
4. `.cursor/context/PROJECT_STATE.md`

---

### ADIM 1 — Mevcut Durumu Anla

Proje şu an bu yapıda:

```
├── cursor/                          ← içindeki dosyaları .cursor/ altına taşı
├── data/
│   ├── processed_data/
│   │   ├── labels/                  ← ham etiket karışıklığı, temizlenecek
│   │   └── mvp_test_v1/             ← TEK ETİKETLİ VERİ SETİ (dokunma, koru)
│   │       ├── test/images + labels
│   │       ├── train/images + labels
│   │       └── valid/images + labels
│   └── raw_data/
│       ├── Photovoltaic system thermography/  ← TERMAL veri (dataset_1, dataset_2)
│       │   ├── dataset_1/annotations + images
│       │   └── dataset_2/annotations + images
│       ├── PV-Multi-Defect-main/             ← RGB, Pascal VOC format (.xml annotations)
│       │   ├── Annotations/
│       │   └── JPEGImages/
│       ├── SolarPanelImagesCleanandFaultyImages/ ← RGB, SINIFLANDIRMA verisi (bbox YOK)
│       │   ├── 640x640/ (bird_drop, Clean, Dusty, Electrical-damage, physical-damage, snow)
│       │   └── Faulty_solar_panel/ (Bird-drop, Clean, Dusty, Electrical-damage...)
│       ├── test/                             ← YOLO formatında etiketli (train/valid/test)
│       │   ├── train/images + labels
│       │   ├── valid/images + labels
│       │   └── test/images + labels
│       ├── Thermal PV Panel Detection Dataset for UAV Inspection/  ← TERMAL YOLO formatı
│       │   ├── train/images + labels
│       │   ├── val/images + labels
│       │   └── test/images + labels
│       └── Thermal Solar PV Anomaly Detection Dataset/            ← TERMAL YOLO formatı
│           └── ImageSet/train + valid + test
├── data.yaml                        ← mevcut YOLO config, içini oku
├── models/v1.0.2/best/             ← mevcut model ağırlıkları
├── reports/v1.0.2/                  ← mevcut raporlar
├── runs/detect/                     ← YOLO eğitim çıktıları (solar_mvp, mvp2, mvp3)
└── scripts/__pycache__/             ← scripts/ klasörü içerikleri muhtemelen boş/eksik
```

---

### ADIM 2 — Yap: Klasör Yapısını Düzenle

Aşağıdaki işlemleri sırayla gerçekleştir:

**2a. `cursor/` klasörünü `.cursor/` ile birleştir**
- Eğer `cursor/` klasöründe dosyalar varsa, bunları `.cursor/` içindeki ilgili konumlara taşı
- `cursor/` klasörünü sil

**2b. `data/` yapısını temizle ve organize et**

Şu anki karmaşık veri yapısını bu hedef yapıya dönüştür:

```
data/
├── raw_data/                        ← SALT-OKUNUR, asla değiştirme
│   ├── rgb/                         ← RGB veri setleri buraya organize et
│   │   ├── PV-Multi-Defect/         ← PV-Multi-Defect-main klasörünü buraya taşı
│   │   │   ├── Annotations/
│   │   │   └── JPEGImages/
│   │   ├── SolarPanel-Classification/ ← SolarPanelImagesCleanandFaultyImages buraya
│   │   │   ├── 640x640/
│   │   │   └── Faulty_solar_panel/
│   │   └── test-yolo-format/        ← raw_data/test/ klasörünü buraya taşı
│   │       ├── train/
│   │       ├── valid/
│   │       └── test/
│   └── thermal/                     ← Termal veri setleri buraya
│       ├── thermography/            ← "Photovoltaic system thermography" buraya
│       │   ├── dataset_1/
│       │   └── dataset_2/
│       ├── UAV-Inspection/          ← "Thermal PV Panel Detection Dataset for UAV Inspection"
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── Anomaly-Detection/       ← "Thermal Solar PV Anomaly Detection Dataset"
│           └── ImageSet/
│
└── processed_data/
    ├── rgb/                         ← RGB eğitim verisi (YOLO formatı)
    │   └── mvp_v1/                  ← mvp_test_v1 klasörünü buraya taşı (isim değişir)
    │       ├── train/images + labels
    │       ├── valid/images + labels
    │       └── test/images + labels
    └── thermal/                     ← Termal eğitim verisi (henüz boş, hazır)
        └── .gitkeep
```

**ÖNEMLİ:** `processed_data/labels/` klasörünü incele — içinde ne varsa raporla, ardından `processed_data/rgb/mvp_v1/` ile çakışmıyorsa `processed_data/rgb/loose_labels/` altına taşı.

**2c. `models/` yapısını düzenle**

```
models/
└── v1.0.2/
    ├── best.pt        ← models/v1.0.2/best/data/best.pt'yi buraya taşı (düzleştir)
    ├── last.pt        ← varsa
    └── metadata.json  ← aşağıda sana şemasını vereceğim, oluştur
```

**2d. `runs/` içini tara ve raporla**
- `runs/detect/solar_mvp/weights/`, `solar_mvp2/weights/`, `solar_mvp3/weights/` içlerindeki `.pt` dosyalarını listele
- En son eğitim hangisi? `solar_mvp3` mü? Son ağırlıkları logla ama taşıma — sadece raporla

**2e. `data.yaml` dosyasını oku ve güncelle**

Mevcut `data.yaml`'ı oku. Sonra içeriğini aşağıdaki hedef yapıya uygun şekilde güncelle:

```yaml
# configs/dataset_rgb.yaml (data.yaml yerine configs/ altında olacak)
path: data/processed_data/rgb/mvp_v1
train: train/images
val:   valid/images
test:  test/images

nc: <mevcut data.yaml'daki nc değerini koru>
names: <mevcut data.yaml'daki names listesini koru>

# Veri seti bilgileri
modality: rgb
version: mvp_v1
notes: "İlk etiketli MVP veri seti. Diğer ham veriler etiketlenmemiş."
```

Ayrıca termal için şablonu da oluştur (içi boş, gelecek için):

```yaml
# configs/dataset_thermal.yaml
path: data/processed_data/thermal/v1
train: train/images
val:   valid/images
test:  test/images

nc: 0        # henüz etiketlenmedi
names: []

modality: thermal
version: v1
notes: "Termal veri seti - etiketleme bekliyor. UAV-Inspection ve Anomaly-Detection ham verileri mevcut."
```

**2f. `configs/` klasörünü oluştur**

```
configs/
├── dataset_rgb.yaml       ← yukarıdaki içerikle
├── dataset_thermal.yaml   ← yukarıdaki şablonla
├── train_local.yaml       ← aşağıda tanımlı (düşük VRAM için)
└── inference.yaml         ← varsayılan eşik değerleri
```

`configs/train_local.yaml` içeriği — **4GB VRAM, Ryzen 7 5800H için optimize**:

```yaml
# Lokal geliştirme profili (4GB VRAM)
model: yolo11n.pt        # En küçük model — lokal için zorunlu
data: configs/dataset_rgb.yaml
imgsz: 416               # 640 yerine 416: VRAM'ı %55 azaltır
epochs: 50               # Test için kısa, gerçek eğitimde 100-150 yap
batch: 4                 # 4GB VRAM için güvenli başlangıç (OOM alırsan 2'ye çek)
workers: 4               # Ryzen 7 için uygun
device: 0                # GPU kullan (cuda:0)
half: false              # FP16 — önce false dene, stabil olunca true yap
cos_lr: true             # Öğrenme hızı düzgün azalsın
patience: 15             # 15 epoch iyileşme yoksa dur
project: runs/detect
name: solar_local

# Bellek kurtarma ayarları
cache: false             # RAM'ı zorlamaz ama eğitim biraz yavaşlar
# NOT: OOM hatası alırsan batch: 2 yap
# NOT: Hâlâ OOM alırsan imgsz: 320 yap
```

`configs/inference.yaml`:

```yaml
conf_threshold: 0.25
iou_threshold: 0.45
max_detections: 300
```

**2g. `scripts/` klasörünü oluştur (boş modül şablonları)**

```
scripts/
├── __init__.py
├── train.py             ← boş şablon (içi TODO yorumu)
├── validate.py          ← boş şablon
├── predict_live.py      ← boş şablon
└── utils/
    ├── __init__.py
    └── versioning.py    ← boş şablon
```

Her boş şablonun içi şu olsun:
```python
"""
<modül adı> — Solar Panel Detection
TODO: Bu modül henüz implement edilmedi.
JOB dosyası: .cursor/jobs/<ilgili JOB>.md
"""
```

---

### ADIM 3 — metadata.json Oluştur

`models/v1.0.2/metadata.json` dosyasını oluştur. Mevcut bilgileri kullan, bilinmeyenleri "unknown" yaz:

```json
{
  "version": "v1.0.2",
  "created_at": "bilinmiyor — runs/detect/ klasöründeki son değişiklik tarihine bak",
  "base_model": "yolo11n.pt",
  "deployment_mode": "local",
  "data_modality": "rgb",
  "dataset": {
    "path": "data/processed_data/rgb/mvp_v1",
    "source": "mvp_test_v1",
    "notes": "Etiketli ilk MVP veri seti"
  },
  "hardware": {
    "gpu": "unknown",
    "vram_gb": 4,
    "ram_gb": 16,
    "cpu": "Ryzen 7 5800H"
  },
  "status": "mevcut — yeniden düzenleme öncesi eğitildi",
  "runs_source": "runs/detect/solar_mvp3"
}
```

---

### ADIM 4 — PROJECT_STATE.md'yi Güncelle

`.cursor/context/PROJECT_STATE.md` dosyasını aç ve mevcut durumu yansıt:

- P1-T1 (iskelet): `✅ DONE` — proje zaten var
- P1-T6 (versiyonlama): `🔄 IN_PROGRESS` — models/v1.0.2 var ama metadata.json yoktu, şimdi oluşturuldu
- Diğerleri: olduğu gibi bırak

---

### ADIM 5 — Durum Raporu Yaz

Tüm işlemler bittikten sonra şu başlıkları içeren kısa bir rapor yaz:

**✅ Yapılanlar:** (madde madde)
**⚠️ Dikkat Edilmesi Gerekenler:** (örn: processed_data/labels/ içinde ne buldun)
**❌ Yapılamadı / Elle Yapılması Gereken:** (varsa)
**📊 Mevcut Veri Durumu:**
  - Etiketli RGB: kaç görüntü (mvp_v1'den)
  - Etiketlenmemiş RGB: hangi klasörler
  - Termal veri: kaç klasör, format ne
**🔜 Sonraki Önerilen Adım:** (hangi JOB başlatılmalı)

---

### SINIR KURALLARI (Bunları Yapma)

- `data/raw_data/` altındaki hiçbir görüntü veya etiketi silme — sadece taşı veya kopyala
- `processed_data/rgb/mvp_v1/` altındaki etiketlere dokunma
- `models/v1.0.2/best.pt` dosyasını silme
- `runs/` klasörünü silme — sadece raporla
- `data.yaml`'ı silme — `configs/dataset_rgb.yaml`'a kopyala, eski yerde bırak (referans için)

---

### TERMAL MODEL SORUSUNUN CEVABI (Ayrıca Bunu da Açıkla)

Kullanıcı termal görüntüler için ayrı bir model eğitmek istiyor. Şunu açıkla:

Termal görüntüler kırmızı renk tabanlı (pseudocolor/false-color) görüntülerdir — gri değil, 3 kanallıdır (RGB gibi). Bu nedenle:

1. **Ayrı model eğitmek doğru yaklaşım** — veri değiştirilir, kod değişmez
2. `configs/dataset_thermal.yaml` hazır — sadece etiketlenmiş termal veri eklenmeli
3. Eğitim komutu neredeyse aynı, sadece `--data configs/dataset_thermal.yaml` değişir
4. Kırmızı renk tabanlı görüntüler YOLO tarafından normal RGB gibi işlenir — özel ayar gerekmez
5. `raw_data/thermal/` altında 3 ayrı termal veri seti mevcut — bunları incele ve hangi formatta olduklarını (YOLO etiketli mi, XML mi, etiketlenmemiş mi) raporla

---

### SUNUMA HAZIRLIK NOTU (Teknik Olmayan Açıklama Şablonu)

Sunum için şu noktaları hazırla — sade dille:

**Model nedir, ne yapar:**
"Bu sistem, güneş panellerinin fotoğraflarına bakarak çatlak, kir, gölge gibi arızaları otomatik olarak işaretleyen bir yapay zeka modelidir. İnsan gözüyle bakıldığında fark edilmesi zor olabilecek sorunları saniyeler içinde tespit eder."

**Nasıl çalışır (3 adımda):**
1. "Fotoğraf sisteme verilir"
2. "Model, panelin her bölgesini tarar"
3. "Arıza tespit edilen bölgelerin etrafına renkli kutu çizer ve türünü yazar"

**Neden YOLO:**
"YOLO, gerçek zamanlı nesne tespiti için endüstri standardı bir yapıdır. Saniyede onlarca görüntüyü işleyebilir."

**Veriler neden önemli:**
"Modelin iyi çalışması için çok sayıda etiketli fotoğraf gerekir. Etiket = her arızanın hangi bölgede olduğunun elle işaretlenmesi. Şu an [X] etiketli görüntümüz var."

**4GB VRAM ile eğitim:**
"Modeli kendi bilgisayarımızda eğitiyoruz. Bilgisayarın grafik kartı (GPU) hesaplamaları yapar. 4GB grafik belleği sınırlı olduğu için küçük model ve düşük çözünürlük kullanıyoruz — bu doğruluk ile hız arasındaki dengedir. Daha güçlü sunucularda daha büyük model çalıştırılabilir."

**Termal kamera farkı:**
"Normal kameralar gözle görünen hasarları tespit eder. Termal kamera ısı dağılımını gösterir — gözle görünmeyen elektrik arızalarını (hotspot) tespit etmek için kullanılır. Her biri için ayrı model eğitiyoruz."

---

Başla. Önce ADIM 0'daki dosyaları oku, sonra sırayla devam et.
