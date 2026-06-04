# 🚀 Aktif Öğrenme (Active Learning) Kullanım Kılavuzu

Bu kılavuz, modelin sahada hata yaptığı (göremediği veya yanlış gördüğü) durumları yakalayarak modeli hızlıca eğitmek (Fine-Tuning) için hazırlanan **2 Fazlı Otomatik Boru Hattının (Pipeline)** nasıl kullanılacağını açıklar.

---

## 🛠️ Ön Hazırlık
Canlı akış (Streaming) ekranındayken modelin zorlandığı veya yanlış tespit yaptığı bir görüntü gördüğünüzde arayüzdeki **"Veri Yakala"** butonuna basın. Bu işlem, o anki kameranın temiz görüntüsünü `data/raw_data/captured/` klasörüne kaydeder. Yeteri kadar görüntü biriktirdiğinizde süreci başlatabilirsiniz.

---

## 🏃‍♂️ Süreci Başlatma

Terminali açın ve projenizin ana dizininde şu komutu çalıştırın:
```bash
python scripts/active_learning_pipeline.py --multiplier 8
```
*(Not: `--multiplier 8` parametresi, onayladığınız her 1 fotoğrafın drone/kamera efektleriyle 8 farklı versiyonunun sentetik olarak üretileceği anlamına gelir.)*

---

## ⚙️ FAZ 1: Otonom Etiketleme ve Doğrulama
Script çalıştığında ilk olarak güncel YOLO modelinizi kullanır ve yakaladığınız görüntülerin üzerine tahminlerini çizer (`.txt` dosyaları oluşturur). Aynı zamanda modelin sınıf isimlerini bir `classes.txt` dosyasına yazar.

Daha sonra script **duraklar** ve sizden etiketleri kontrol etmenizi bekler.

### 📝 Label Studio ile Kontrol (Senin Görevin)
Script durakladığında arka planda `uv run label-studio` komutunu tetikler. Tarayıcınızda Label Studio açıldığında şu adımları izleyin:
1. Yeni bir proje oluşturun ve Data Import (Veri Ekleme) ekranına gelin.
2. `data/raw_data/captured/` klasöründeki resimleri ve **`classes.txt`** dosyasını projeye dahil edin. *(classes.txt'yi dahil etmek çok önemlidir, böylece modelinizin 0, 1 gibi class numaraları isimlerle doğru eşleşir).*
3. Resimler arasında gezinin. Modelin "yanlış çizdiği" kutuları silin veya "göremediği" nesneleri kendiniz çizin.
4. Kaydedin.

---

## 🧬 FAZ 2: Çoğaltma ve Veri Setini Hazırlama
Label Studio'da hataları düzeltip işinizi bitirdikten sonra terminale geri dönün ve **ENTER** tuşuna basın.

Bundan sonra script kontrolü tekrar devralır:
1. **Güvenlik Duvarı (Validation):** Tüm `.txt` dosyalarını tarar. Eğer yanlışlıkla modelde olmayan bir sınıf numarası girdiyseniz (Örn: Model 2 sınıflı iken 3 numaralı sınıf girilmişse), o dosyayı tespit edip iptal eder. Sistemi çökmekten kurtarır.
2. **Çoğaltma (Augmentation):** Hatasız olan tüm verileri alır, belirttiğimiz `multiplier` (örneğin 8 katı) kadar bulanıklık, titreme, karıncalanma, uzaklaştırma gibi fiziksel zorluklar ekleyerek çoğaltır.
3. **Dağıtım (Merge):** Orijinal ve çoğaltılmış tüm verileri karıştırır. %80 Eğitim (Train), %20 Test (Val) olacak şekilde `data/processed/finetune` klasörüne yerleştirir.
4. **Temizlik:** Yakalanan ham veriler `archive` klasörüne yedeklenerek ortalık temizlenir.

---

## 🎯 Son Adım: Modeli Eğitme
Artık `data/processed/finetune` klasörünüzde tamamen kusursuz ve çoğaltılmış bir veri seti var. Tek yapmanız gereken her zamanki eğitim komutunuzu çalıştırarak modeli bu yeni verilerle beslemektir. Başarılar!
