# PROJECT STATE — Solar Panel Streaming Alt Projesi
# ============================================================
# TEK GERÇEK KAYNAK — Her JOB sadece kendi satırını günceller.
# ============================================================

**Son Güncelleme:** 2026-04-13
**Aktif Faz:** Phase 1 — Temel Pipeline
**Genel İlerleme:** 0 / 7 görev tamamlandı

---

## PHASE 1 — Core Pipeline
> **Hedef:** Çalışan end-to-end video inference + kayıt pipeline'ı
> **Paralel Çalışabilir:** S-T1, S-T3, S-T7 birlikte başlayabilir

| ID | Görev | Durum | Branch | JOB Dosyası |
|---|---|---|---|---|
| S-T1 | Model Loader (`model_loader.py`) | ⬜ TODO | `streaming/job/S-T1-model-loader` | `.cursor/jobs/S-T1.md` |
| S-T2 | Frame Processor (`frame_processor.py`) | ⬜ BLOCKED (S-T1) | `streaming/job/S-T2-processor` | `.cursor/jobs/S-T2.md` |
| S-T3 | Video Source Manager (`source_manager.py`) | ⬜ TODO | `streaming/job/S-T3-source` | `.cursor/jobs/S-T3.md` |
| S-T4 | Annotator / Overlay (`annotator.py`) | ⬜ BLOCKED (S-T2) | `streaming/job/S-T4-annotator` | `.cursor/jobs/S-T4.md` |
| S-T5 | Video Recorder (`recorder.py`) | ⬜ BLOCKED (S-T4) | `streaming/job/S-T5-recorder` | `.cursor/jobs/S-T5.md` |
| S-T6 | UI / Kontrol Paneli (`control_panel.py`) | ⬜ BLOCKED (S-T1..S-T5) | `streaming/job/S-T6-ui` | `.cursor/jobs/S-T6.md` |
| S-T7 | Config & CLI & `main.py` | ⬜ TODO | `streaming/job/S-T7-config` | `.cursor/jobs/S-T7.md` |

---

## DURUM KODLARI
- ⬜ `TODO` — Başlanmadı
- 🔒 `BLOCKED` — Bağımlılık bekliyor
- 🔄 `IN_PROGRESS` — Aktif
- ✅ `DONE` — Tamamlandı, test edildi
- ❌ `FAILED` — Test başarısız
