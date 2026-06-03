#!/usr/bin/env python3
"""
Thermal Anomaly-Detection veri setini 5 sınıfa indirgeyip processed_data'ya kopyalar.

Orijinal 8 sınıf → hedef 5 sınıf:
  0 MultiByPassed          → 0 (ByPassed)
  1 MultiDiode             → 1 (Diode)
  2 MultiHotSpot           → 2 (HotSpot)
  3 SingleByPassed         → 0 (ByPassed)
  4 SingleDiode            → 1 (Diode)
  5 SingleHotSpot          → 2 (HotSpot)
  6 StringOpenCircuit      → 3 (StringOpenCircuit)
  7 StringReversedPolarity → 4 (StringReversedPolarity)

Not: Hiçbir görüntü veya annotation silinmez; sadece class id'leri yeniden numaralandırılır.

Kullanım:
  uv run scripts/filter_thermal_labels.py
  # ya da
  python scripts/filter_thermal_labels.py [--src ...] [--dst ...]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Haritalama: eski_id -> yeni_id  (None = sil / şu an None yok, hepsi tutulur)
CLASS_MAP: dict[int, int | None] = {
    0: 0,   # MultiByPassed          → ByPassed
    1: 1,   # MultiDiode             → Diode
    2: 2,   # MultiHotSpot           → HotSpot
    3: 0,   # SingleByPassed         → ByPassed
    4: 1,   # SingleDiode            → Diode
    5: 2,   # SingleHotSpot          → HotSpot
    6: 3,   # StringOpenCircuit      → StringOpenCircuit
    7: 4,   # StringReversedPolarity → StringReversedPolarity
}

NEW_NAMES = ["ByPassed", "Diode", "HotSpot", "StringOpenCircuit", "StringReversedPolarity"]
SPLITS = ["train", "valid", "test"]


def remap_label_file(src: Path, dst: Path) -> tuple[int, int]:
    """
    Tek bir YOLO label dosyasını filtreler/remaplar.
    Returns (kept_lines, dropped_lines).
    """
    lines = src.read_text().splitlines()
    kept, dropped = [], 0
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(parts[0])
        new_id = CLASS_MAP.get(cls_id)
        if new_id is None:
            dropped += 1
            continue
        kept.append(f"{new_id} " + " ".join(parts[1:]))

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept) + ("\n" if kept else ""))
    return len(kept), dropped


def copy_images(src_img_dir: Path, dst_img_dir: Path) -> int:
    """Görüntüleri doğrudan kopyalar (değişiklik gerekmez)."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in src_img_dir.glob("*"):
        if img.is_file():
            shutil.copy2(img, dst_img_dir / img.name)
            count += 1
    return count


def write_data_yaml(dst_root: Path) -> None:
    yaml_content = f"""# Termal Anomaly-Detection - 5 sınıf
# Oluşturuldu: filter_thermal_labels.py
path: {dst_root.resolve()}
train: train/images
val:   valid/images
test:  test/images

nc: {len(NEW_NAMES)}
names: {NEW_NAMES}

# Kaynak: raw_data/thermal/Anomaly-Detection (8 sınıf)
# Birleştirme: MultiByPassed+SingleByPassed→ByPassed
#              MultiDiode+SingleDiode→Diode
#              MultiHotSpot+SingleHotSpot→HotSpot
# Korunan (birebir): StringOpenCircuit, StringReversedPolarity
"""
    (dst_root / "data.yaml").write_text(yaml_content)
    print(f"  ✔ data.yaml yazıldı → {dst_root / 'data.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thermal label remapping to 5 classes")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/raw_data/thermal/Anomaly-Detection/ImageSet"),
        help="Kaynak ImageSet dizini",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/processed_data/thermal_anomaly_v1"),
        help="Hedef işlenmiş dataset dizini",
    )
    args = parser.parse_args()

    src: Path = args.src
    dst: Path = args.dst

    if not src.exists():
        raise FileNotFoundError(f"Kaynak bulunamadı: {src}")

    if dst.exists():
        print(f"  ⚠ Hedef dizin zaten var, üzerine yazılacak: {dst}")

    total_kept = total_dropped = total_imgs = 0

    for split in SPLITS:
        src_lbl = src / split / "labels"
        src_img = src / split / "images"
        dst_lbl = dst / split / "labels"
        dst_img = dst / split / "images"

        if not src_lbl.exists():
            print(f"  ⚠ {split}/labels yok, atlanıyor")
            continue

        # Görüntüleri kopyala
        n_imgs = copy_images(src_img, dst_img) if src_img.exists() else 0
        total_imgs += n_imgs

        # Labelları remap et
        label_files = list(src_lbl.glob("*.txt"))
        split_kept = split_dropped = 0
        for lf in label_files:
            k, d = remap_label_file(lf, dst_lbl / lf.name)
            split_kept += k
            split_dropped += d

        total_kept += split_kept
        total_dropped += split_dropped
        print(
            f"  [{split:6s}] {n_imgs:5d} görüntü | "
            f"{split_kept:6d} annotation tutuldu | "
            f"{split_dropped:5d} annotation silindi"
        )

    write_data_yaml(dst)

    print("\n  ===== ÖZET =====")
    print(f"  Toplam görüntü kopyalandı : {total_imgs}")
    print(f"  Toplam annotation tutuldu  : {total_kept}")
    print(f"  Toplam annotation silindi  : {total_dropped}")
    print(f"  Hedef sınıflar            : {NEW_NAMES}")
    print(f"  Çıktı dizini              : {dst.resolve()}")
    print("\n  Sonraki adım:")
    print("  python scripts/train.py --config configs/model_thermal.yaml --data-config data/processed_data/thermal_anomaly_v1/data.yaml")


if __name__ == "__main__":
    main()
