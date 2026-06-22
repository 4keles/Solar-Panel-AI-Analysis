#!/usr/bin/env python3
"""
Otomatik eğitim döngüsü.

Adımlar:
  1. Format1 augmentasyon (augment_format1.py)
  2. Round-1: Tam eğitim (yolo11s, augmented dataset)
  3. Metrik kontrolü → fine-tune gerekiyor mu?
  4. Round-2: Fine-tune (frozen backbone, düşük LR) — gerekirse
  5. Final rapor (generate_report.py)

Usage:
  python scripts/auto_loop.py [--skip-aug]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
VENV   = ROOT / ".venv/bin/python"
LOGS   = ROOT / "logs"
MODELS = ROOT / "models"

LOGS.mkdir(exist_ok=True)

MAP50_GOOD      = 0.72   # bu eşiğin üstü = Round-2 gerekmez
MAP50_MIN       = 0.50   # bu altı = ciddi sorun var, yine de fine-tune dene
FINETUNE_FREEZE = 10     # backbone layer sayısı


def run(cmd: list[str], log_path: Path, label: str) -> int:
    """Komutu çalıştır, stdout+stderr'i log_path'e yaz, canlı da göster."""
    print(f"\n{'='*60}")
    print(f"  [{label}] Başlıyor…")
    print(f"  Log: {log_path}")
    print(f"{'='*60}\n")

    with open(log_path, "w") as fout:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(ROOT),
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            fout.write(line)
        proc.wait()

    print(f"\n[{label}] Tamamlandı — exit code: {proc.returncode}")
    return proc.returncode


def latest_version() -> str | None:
    import re
    dirs = [d.name for d in MODELS.iterdir()
            if d.is_dir() and re.match(r"v\d+\.\d+\.\d+", d.name)]
    if not dirs:
        return None
    return sorted(dirs, key=lambda s: tuple(int(x) for x in s[1:].split(".")))[-1]


def read_summary(version: str) -> dict | None:
    p = ROOT / "reports" / version / "val_summary.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def print_metrics(summary: dict) -> None:
    o = summary["overall"]
    print(f"\n  mAP50={o['mAP50']:.3f}  mAP50-95={o['mAP50_95']:.3f}"
          f"  P={o['precision']:.3f}  R={o['recall']:.3f}  F1={o['f1']:.3f}")
    if summary.get("per_class"):
        print(f"  {'Class':<22} {'mAP50':>6}  {'F1':>6}  {'R':>6}")
        for cls, m in sorted(summary["per_class"].items(),
                             key=lambda x: x[1]["mAP50"]):
            print(f"  {cls:<22} {m['mAP50']:>6.3f}  {m['f1']:>6.3f}  {m['recall']:>6.3f}")


def generate_report(version: str, run_dir: Path) -> None:
    run([
        str(VENV), "scripts/generate_report.py",
        "--weights",  str(MODELS / version / "best.pt"),
        "--run-dir",  str(run_dir),
        "--data",     "configs/dataset_unifiedV2.yaml",
        "--version",  version,
        "--split",    "test",
        "--device",   "0",
    ], LOGS / f"report_{version}.log", f"Rapor {version}")


def find_run_dir(name: str) -> Path | None:
    """solar_v1.0.5_rgb gibi isimle run dizinini bul."""
    for p in ROOT.rglob(f"{name}/args.yaml"):
        return p.parent
    return None


def main(skip_aug: bool = False) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")

    # ── Step 1: Augmentasyon ─────────────────────────────────────────────────
    if not skip_aug:
        rc = run(
            [str(VENV), "scripts/augment_format1.py", "--multiplier", "5"],
            LOGS / f"aug_{ts}.log",
            "Format1 Augmentasyon (5x)",
        )
        if rc != 0:
            print("HATA: Augmentasyon başarısız. Devam ediliyor (aug olmadan)…")
    else:
        print("\n[--skip-aug] Augmentasyon atlandı.")

    # ── Step 2: Round-1 Eğitim ───────────────────────────────────────────────
    rc = run([
        str(VENV), "scripts/train.py",
        "--config",      "configs/train_v1.0.5_rgb.yaml",
        "--data-config", "configs/dataset_unifiedV2.yaml",
        "--version-bump", "minor",
    ], LOGS / f"train_r1_{ts}.log", "Round-1 Eğitim (yolo11s, augmented)")

    if rc != 0:
        print("HATA: Round-1 eğitim başarısız. Çıkılıyor.")
        sys.exit(1)

    r1_version = latest_version()
    r1_run_dir = find_run_dir("solar_v1.0.5_rgb")
    print(f"\nRound-1 model: {r1_version}  |  run: {r1_run_dir}")

    # ── Rapor + Metrik okuma ─────────────────────────────────────────────────
    if r1_run_dir:
        generate_report(r1_version, r1_run_dir)
    summary_r1 = read_summary(r1_version)
    if summary_r1:
        print("\n=== Round-1 Test Metrikleri ===")
        print_metrics(summary_r1)
        map50_r1 = summary_r1["overall"]["mAP50"]
    else:
        print("Uyarı: Round-1 metrik okunamadı, fine-tune yapılacak.")
        map50_r1 = 0.0

    # ── Sorunlu sınıfları tespit et ──────────────────────────────────────────
    weak_classes = []
    if summary_r1 and summary_r1.get("per_class"):
        for cls, m in summary_r1["per_class"].items():
            if m["mAP50"] < 0.5:
                weak_classes.append(cls)
        if weak_classes:
            print(f"\nDüşük performanslı sınıflar: {weak_classes}")

    # ── Step 3: Fine-tune karar ───────────────────────────────────────────────
    if map50_r1 >= MAP50_GOOD:
        print(f"\nmAP50={map50_r1:.3f} ≥ {MAP50_GOOD} — Fine-tune gerekmez. Bitti.")
        print(f"\nRapor: file://{ROOT}/reports/{r1_version}/report.html")
        return

    print(f"\nmAP50={map50_r1:.3f} < {MAP50_GOOD} — Round-2 fine-tune başlatılıyor…")

    # Fine-tune config oluştur
    finetune_cfg = ROOT / "configs" / "train_finetune_r2.yaml"
    # copy_paste artır, özellikle bird_drop için
    cp_val = 0.6 if "bird_drop" in weak_classes else 0.4
    finetune_cfg.write_text(f"""# Round-2 Fine-tune — otomatik üretildi
model: {MODELS}/{r1_version}/best.pt
data: configs/dataset_unifiedV2.yaml
imgsz: 640
epochs: 100
batch: 8
workers: 4
device: 0
half: false
cos_lr: true
patience: 25
project: runs
name: solar_v1.0.5_finetune
cache: false

# Backbone donduruldu — sadece head eğitilir
# (freeze: train.py --freeze argümanıyla verilir)

# Augmentation — azalt, overfit önle
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.3
degrees: 10.0
translate: 0.1
scale: 0.4
flipud: 0.3
fliplr: 0.5
mosaic: 0.8
mixup: 0.1
copy_paste: {cp_val}
erasing: 0.3
auto_augment: randaugment

cls: 2.0
label_smoothing: 0.05
warmup_epochs: 2.0
weight_decay: 0.0005
""")
    print(f"Fine-tune config: {finetune_cfg}")

    # Round-2
    rc = run([
        str(VENV), "scripts/train.py",
        "--config",       str(finetune_cfg),
        "--data-config",  "configs/dataset_unifiedV2.yaml",
        "--finetune",     str(MODELS / r1_version / "best.pt"),
        "--freeze",       str(FINETUNE_FREEZE),
        "--version-bump", "patch",
    ], LOGS / f"train_r2_{ts}.log", "Round-2 Fine-tune (frozen backbone)")

    if rc != 0:
        print("HATA: Round-2 başarısız. Round-1 sonucu kullanılıyor.")
        print(f"Rapor: file://{ROOT}/reports/{r1_version}/report.html")
        return

    r2_version = latest_version()
    r2_run_dir = find_run_dir("solar_v1.0.5_finetune")
    print(f"\nRound-2 model: {r2_version}  |  run: {r2_run_dir}")

    if r2_run_dir:
        generate_report(r2_version, r2_run_dir)
    summary_r2 = read_summary(r2_version)

    # ── Final karşılaştırma ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL SONUÇ KARŞILAŞTIRMASI")
    print("="*60)
    if summary_r1:
        print(f"\nRound-1 ({r1_version}):")
        print_metrics(summary_r1)
    if summary_r2:
        print(f"\nRound-2 Fine-tune ({r2_version}):")
        print_metrics(summary_r2)
        map50_r2 = summary_r2["overall"]["mAP50"]
        delta = map50_r2 - map50_r1
        sign  = "+" if delta >= 0 else ""
        print(f"\n  Fine-tune etkisi: mAP50 {map50_r1:.3f} → {map50_r2:.3f} ({sign}{delta:.3f})")

    best = r2_version if (summary_r2 and summary_r2["overall"]["mAP50"] > map50_r1) else r1_version
    print(f"\n  En iyi model: {best}")
    print(f"  Rapor:  file://{ROOT}/reports/{best}/report.html")
    print(f"  Weights: {ROOT}/models/{best}/best.pt")
    print("="*60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-aug", action="store_true",
                    help="Augmentasyon adımını atla (daha önce yapıldıysa)")
    args = ap.parse_args()
    main(skip_aug=args.skip_aug)
