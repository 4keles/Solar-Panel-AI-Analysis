# KULLANIM KILAVUZU — Cursor Mikro-JOB Sistemi

## Bu Sistem Ne Yapar?

Projeyi **bağımsız, paralel çalışabilir mikro-görevlere (JOB)** böler. Her JOB:
- Tam olarak ne OKUYACAĞINI bilir (token tasarrufu)
- Tam olarak ne YAZACAĞINI bilir (çakışma yok)
- Kendi testini çalıştırır
- Git commit ile kapanır

---

## HIZLI BAŞLANGIÇ (5 Adım)

### Adım 1 — Cursor'da Projeyi Aç

```bash
cd solar-panel-detection
cursor .
```

### Adım 2 — Aktif JOB'u Belirle

`.cursor/context/PROJECT_STATE.md` dosyasına bak. `⬜ TODO` durumundaki ve bağımlılıkları karşılanmış JOB'u seç.

```
Örnek: Şu an P1-T1, P1-T2, P1-T3 bağımlılıksız başlayabilir.
       3 ayrı Cursor oturumunda paralel çalıştırabilirsin.
```

### Adım 3 — Cursor Chat'e Tam Prompt Ver

```
Aşağıdaki JOB dosyasını oku ve uygula:
.cursor/jobs/P1-T2.md

Başlamadan önce:
1. PROJECT_STATE.md'yi oku (hangi aşamadayız)
2. JOB dosyasındaki READS bölümündeki dosyaları oku
3. Başka hiçbir dosyayı okuma
```

### Adım 4 — JOB Tamamlandığında

Agent self-test komutlarını çalıştıracak. Başarılıysa:

```bash
# Agent şunu çalıştırır:
git checkout -b job/P1-T2-infra
git add scripts/utils/logger.py ...
git commit -m "P1-T2: Logger, ConfigLoader ve Device yardımcıları ✅"
```

### Adım 5 — PROJECT_STATE.md Güncelle

```markdown
<!-- Şunu değiştir: -->
| P1-T2 | Logger + Config sistemi | ⬜ TODO |

<!-- Buna: -->
| P1-T2 | Logger + Config sistemi | ✅ DONE |
```

---

## PARALEL ÇALIŞMA AKIŞI

```
Terminal 1 (Cursor oturumu A):     Terminal 2 (Cursor oturumu B):
   JOB: P1-T1-skeleton                JOB: P1-T2-infra
   branch: job/P1-T1-skeleton         branch: job/P1-T2-infra
   Yazıyor: pyproject.toml            Yazıyor: logger.py, config_loader.py
   
Terminal 3 (Cursor oturumu C):
   JOB: P1-T3-converter
   branch: job/P1-T3-converter
   Yazıyor: converters.py
```

Üç JOB birbirinin dosyasına dokunmaz → merge çakışması yok.

---

## ÇAKIŞMA ÇÖZME

Bir JOB çakışma tespit ederse şu formatı kullan:

```
⚠️ ÇAKIŞMA TESPİT EDİLDİ
JOB: P1-T4 (dataset_prep.py)
Çakışan JOB: P1-T3 (converters.py)
Çakışma: converters.py'deki BoundingBox sınıfı — ben de kullanmak istiyorum
Öneri: P1-T3 tamamlanana kadar bekle, sonra import et
```

Çözüm: `.cursor/context/CONTRACTS.md`'yi oku. Her modülün sahibi orada yazıyor.

---

## TOKEN TASARRUFU KURALLARI

| Yapma | Yap |
|---|---|
| `architecture.md` her seferinde oku | `ARCH_SUMMARY.md` oku (1/10 boyutu) |
| Tüm scripts/ klasörünü tara | Sadece JOB'daki READS dosyalarını oku |
| Test geçmeden commit yap | SELF TEST bölümünü çalıştır, sonra commit |
| Bir JOB'da 2 modül yaz | Her JOB = 1 modül + kendi testi |

---

## GIT BRANCH STRATEJİSİ

```bash
# Her JOB kendi branch'inde:
git checkout -b job/P1-T2-infra

# JOB tamamlanınca phase branch'ine merge:
git checkout phase-1
git merge job/P1-T2-infra --no-ff
git push

# Tüm phase tamamlanınca main'e:
git checkout main
git merge phase-1 --no-ff
git tag v1.0.0-phase1
```

---

## YENİ JOB OLUŞTURMA

Planlama dışında yeni bir görev çıkarsa:

1. `.cursor/jobs/_TEMPLATE.md`'yi kopyala
2. JOB ID belirle: `PX-TY-kisa-aciklama`
3. `PROJECT_STATE.md`'ye yeni satır ekle
4. `CONTRACTS.md`'ye yeni arayüz sözleşmesi ekle (varsa)
5. Bağımlılıklarını tanımla

---

## BAĞIMLILIK GÖRSELİ

```
P1-T1 (iskelet) ──┬──→ P1-T2 (logger/config) ──┬──→ P2-T1 (train config)
                  │                              │
                  ├──→ P1-T3 (converter)  ───→ P1-T4 (dataset_prep) ──→ P1-T5 (augment)
                  │                              
                  └──→ P1-T6 (versioning) ──────→ P2-T5 (promoter)
                                                         │
                         P2-T2 (trainer) ───────────────┘
                              │
                    ┌─────────┼──────────┐
                   P2-T3    P2-T4      P2-T6
                 (resume) (finetune) (metadata)
                              │
                         P3-T1 (validator) ──→ P3-T2 (reporter)
                              │                      │
                         P3-T3 (regression)    P3-T4 (benchmark)
                              │
                    ┌─────────┴──────────┐
                P4-T1 (inference)    P4-T2 (export)
                    │
              P4-T3 (rtsp) → P4-T4 (edge deploy) → P4-T5 (CI/CD)
```

---

## SORUN GİDERME

**"Hangi JOB'u başlatmalıyım?"**
→ `PROJECT_STATE.md` aç, `⬜ TODO` olan ve `BLOCKED` olmayan ilkini seç

**"Bu fonksiyonu hangi JOB yazacak?"**
→ `CONTRACTS.md`'de sahip JOB yazıyor

**"Test çalışmıyor, bağımlılık modülü henüz yok"**
→ `conftest.py`'daki mock fixture'ları kullan, gerçek modülü mock'la

**"Yeni bir bağımlılığa ihtiyacım var"**
→ `pyproject.toml`'a dokunma, yeni `INFRA` JOB'u aç: `PX-INFRA-new-dep.md`

**"İki agent aynı dosyayı düzenliyor"**
→ Çakışma formatını kullan, `CONTRACTS.md`'de sahip JOB'u bul, diğerini blokla
