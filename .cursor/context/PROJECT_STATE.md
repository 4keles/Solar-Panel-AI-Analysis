# PROJECT STATE — Solar Panel Object Detection
# ============================================================
# Bu dosya projenin TEK GERÇEK KAYNAĞI (single source of truth)'dır.
# Her JOB tamamlandığında sadece kendi satırını ✅ DONE yapar.
# Başka hiçbir şeyi değiştirme.
# ============================================================

**Son Güncelleme:** 2025-01-15  
**Aktif Faz:** Phase 1  
**Genel İlerleme:** 0 / 23 görev tamamlandı

---

## PHASE 1 — Altyapı ve Veri Katmanı
> **Hedef:** Çalışan veri hazırlama pipeline'ı + test edilmiş proje iskeleti  
> **Bağımlılık:** Yok (başlangıç fazı)  
> **Paralel Çalışabilir:** P1-T1, P1-T2, P1-T3 birlikte başlayabilir

| ID | Görev | Durum | Branch | JOB Dosyası |
|---|---|---|---|---|
| P1-T1 | Proje iskelet yapısı + uv kurulum | ✅ DONE | `job/P1-T1-skeleton` | `.cursor/jobs/P1-T1.md` |
| P1-T2 | Logger + Config sistemi | ⬜ TODO | `job/P1-T2-infra` | `.cursor/jobs/P1-T2.md` |
| P1-T3 | Ham veri → YOLO format dönüştürücü (`unify_classes.py`) | ✅ DONE | `job/P1-T3-converter` | `.cursor/jobs/P1-T3.md` |
| P1-T4 | Dataset hazırlama CLI (`dataset_prep.py`) | ⬜ BLOCKED (P1-T3) | `job/P1-T4-dataprep` | `.cursor/jobs/P1-T4.md` |
| P1-T5 | Augmentation pipeline (`augment.py`) | ⬜ BLOCKED (P1-T4) | `job/P1-T5-augment` | `.cursor/jobs/P1-T5.md` |
| P1-T6 | Versiyonlama sistemi (`versioning.py`) | 🔄 IN_PROGRESS | `job/P1-T6-versioning` | `.cursor/jobs/P1-T6.md` |

---

## PHASE 2 — Eğitim Katmanı
> **Hedef:** Tam çalışan eğitim pipeline'ı (edge + host modu, resume, finetune)  
> **Bağımlılık:** Phase 1 tamamlanmalı  
> **Paralel Çalışabilir:** P2-T1 ve P2-T2 birlikte; P2-T3 ve P2-T4 birlikte

| ID | Görev | Durum | Branch | JOB Dosyası |
|---|---|---|---|---|
| P2-T1 | Eğitim konfigürasyon sistemi | ⬜ BLOCKED (P1) | `job/P2-T1-train-config` | `.cursor/jobs/P2-T1.md` |
| P2-T2 | YOLO training wrapper (`train.py`) | ⬜ BLOCKED (P2-T1) | `job/P2-T2-trainer` | `.cursor/jobs/P2-T2.md` |
| P2-T3 | Resume/Checkpoint kurtarma | ⬜ BLOCKED (P2-T2) | `job/P2-T3-resume` | `.cursor/jobs/P2-T3.md` |
| P2-T4 | Transfer learning & freeze sistemi | ⬜ BLOCKED (P2-T2) | `job/P2-T4-finetune` | `.cursor/jobs/P2-T4.md` |
| P2-T5 | Post-training artifact promoter | ⬜ BLOCKED (P2-T2, P1-T6) | `job/P2-T5-promoter` | `.cursor/jobs/P2-T5.md` |
| P2-T6 | Metadata sistemi (`metadata.py`) | ⬜ BLOCKED (P2-T5) | `job/P2-T6-metadata` | `.cursor/jobs/P2-T6.md` |

---

## PHASE 3 — Doğrulama ve Raporlama Katmanı
> **Hedef:** Otomatik model doğrulama, regresyon testi, görsel raporlar  
> **Bağımlılık:** Phase 2 tamamlanmalı  
> **Paralel Çalışabilir:** P3-T1 ve P3-T2 birlikte

| ID | Görev | Durum | Branch | JOB Dosyası |
|---|---|---|---|---|
| P3-T1 | Doğrulama motoru (`validate.py`) | ⬜ BLOCKED (P2) | `job/P3-T1-validator` | `.cursor/jobs/P3-T1.md` |
| P3-T2 | Rapor üretici (grafikler, JSON özet) | ⬜ BLOCKED (P3-T1) | `job/P3-T2-reporter` | `.cursor/jobs/P3-T2.md` |
| P3-T3 | Model regresyon test sistemi | ⬜ BLOCKED (P3-T2) | `job/P3-T3-regression` | `.cursor/jobs/P3-T3.md` |
| P3-T4 | Benchmark modülü (`benchmark.py`) | ⬜ BLOCKED (P3-T1) | `job/P3-T4-benchmark` | `.cursor/jobs/P3-T4.md` |

---

## PHASE 4 — Çıkarım ve Dağıtım Katmanı
> **Hedef:** Gerçek zamanlı çıkarım, model dışa aktarma, edge/host dağıtım araçları  
> **Bağımlılık:** Phase 3 tamamlanmalı  
> **Paralel Çalışabilir:** P4-T1, P4-T2, P4-T3 birlikte

| ID | Görev | Durum | Branch | JOB Dosyası |
|---|---|---|---|---|
| P4-T1 | Canlı çıkarım motoru (`predict_live.py`) | ⬜ BLOCKED (P3) | `job/P4-T1-inference` | `.cursor/jobs/P4-T1.md` |
| P4-T2 | Model dışa aktarma (`export_model.py`) | ⬜ BLOCKED (P3) | `job/P4-T2-export` | `.cursor/jobs/P4-T2.md` |
| P4-T3 | RTSP stream yöneticisi | ⬜ BLOCKED (P4-T1) | `job/P4-T3-rtsp` | `.cursor/jobs/P4-T3.md` |
| P4-T4 | Edge deployment paketi (Jetson/RPi) | ⬜ BLOCKED (P4-T2) | `job/P4-T4-edge-deploy` | `.cursor/jobs/P4-T4.md` |
| P4-T5 | CI/CD workflow'ları (.github/workflows) | ⬜ BLOCKED (P3-T3) | `job/P4-T5-cicd` | `.cursor/jobs/P4-T5.md` |

---

## DURUM KODLARI
- ⬜ `TODO` — Başlanmadı, bağımlılıklar karşılandı
- 🔒 `BLOCKED` — Bağımlılık tamamlanmadı
- 🔄 `IN_PROGRESS` — Aktif olarak çalışılıyor
- ✅ `DONE` — Tamamlandı, test edildi, commit edildi
- ❌ `FAILED` — Test başarısız, yeniden yapılacak
