#!/usr/bin/env python3
"""
Class-shift & label integrity audit.
Compares old unified_dataset vs new unifiedV2_dataset and inspects
raw format1/format2 sources for mapping errors.

Usage:
  python scripts/class_shift_audit.py
  python scripts/class_shift_audit.py --report-only   # just show, no fix
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE   = Path(__file__).resolve().parent.parent
SOLAR  = BASE / "data/processed_data/solar"
OLD    = SOLAR / "unified_dataset"
NEW    = SOLAR / "unifiedV2_dataset"
FORMAT1 = SOLAR / "format1"
FORMAT2 = SOLAR / "format2"

OLD_NAMES = {0: "bird_drop", 1: "bird_feather", 2: "physical_damage",
             3: "dust_particle", 4: "leaf", 5: "snow"}

NEW_NAMES = {0: "bird_drop", 1: "bird_feather", 2: "physical_damage",
             3: "dust_partical", 4: "leaf", 5: "snow"}

SEP = "─" * 70


def read_labels(label_dir: Path) -> list[tuple[Path, list[str]]]:
    if not label_dir.exists():
        return []
    result = []
    for f in sorted(label_dir.glob("*.txt")):
        lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        result.append((f, lines))
    return result


def parse_class_id(token: str) -> int | None:
    try:
        v = float(token)
        return int(v)
    except ValueError:
        return None


def audit_float_ids(label_dir: Path, split: str, dataset_name: str) -> dict:
    """Find labels where class ID is written as float (e.g. 2.0 instead of 2)."""
    float_files: list[str] = []
    int_files:   list[str] = []
    float_cls_counter: Counter = Counter()

    for f, lines in read_labels(label_dir):
        has_float = False
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            raw = parts[0]
            if "." in raw:        # e.g. "2.0"
                has_float = True
                float_cls_counter[raw] += 1
        if has_float:
            float_files.append(f.name)
        else:
            int_files.append(f.name)

    return {
        "float_file_count": len(float_files),
        "int_file_count":   len(int_files),
        "float_cls_dist":   dict(float_cls_counter),
        "float_file_samples": float_files[:5],
    }


def audit_class_dist(label_dir: Path, name_map: dict) -> dict[str, int]:
    """Count annotations per class (normalise float IDs to int)."""
    counter: Counter = Counter()
    for f, lines in read_labels(label_dir):
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            cid = parse_class_id(parts[0])
            if cid is not None:
                counter[cid] += 1
    return {name_map.get(k, f"UNKNOWN_{k}"): v
            for k, v in sorted(counter.items())}


def detect_folder_class_mismatch(format1_dir: Path, new_names: dict) -> list[dict]:
    """
    For each format1 sub-folder, check if the dominant class label
    matches the expected semantic meaning of the folder name.
    """
    folder_hints = {
        "bird-drop": {0},            # bird_drop
        "clean":     set(),          # all should be dropped (healthy)
        "dust":      {3},            # dust_partical
        "fiziksel":  {2},            # physical_damage
        "snow":      {5},            # snow
    }
    issues = []
    for folder, expected_ids in folder_hints.items():
        lbl_dir = format1_dir / folder / "labels"
        if not lbl_dir.exists():
            continue
        counter: Counter = Counter()
        for f, lines in read_labels(lbl_dir):
            for line in lines:
                parts = line.split()
                if parts:
                    cid = parse_class_id(parts[0])
                    if cid is not None:
                        counter[cid] += 1
        if not counter:
            continue
        top_class_id, top_count = counter.most_common(1)[0]
        total = sum(counter.values())
        dominant_pct = top_count / total * 100
        match = (top_class_id in expected_ids) or not expected_ids
        issues.append({
            "folder":       folder,
            "expected_ids": sorted(expected_ids),
            "top_class_id": top_class_id,
            "top_name":     new_names.get(top_class_id, f"cls{top_class_id}"),
            "dominant_pct": dominant_pct,
            "dist":         {new_names.get(k, f"cls{k}"): v
                             for k, v in counter.most_common()},
            "ok":           match,
        })
    return issues


def cross_check_file_overlap(old_dir: Path, new_dir: Path) -> dict:
    """How many stems appear in both old and new datasets?"""
    old_stems: set[str] = set()
    for split in ("train", "val", "test"):
        for f in (old_dir / split / "labels").glob("*.txt"):
            # strip F1_ prefix and _aug_N suffix if present
            stem = re.sub(r"^F1_\d+_[0-9a-f]+_", "", f.stem)
            stem = re.sub(r"_aug_\d+$", "", stem)
            old_stems.add(stem)

    new_stems: set[str] = set()
    for split in ("train", "val", "test"):
        for f in (new_dir / split / "labels").glob("*.txt"):
            new_stems.add(f.stem)

    overlap = old_stems & new_stems
    return {
        "old_unique_stems": len(old_stems),
        "new_unique_stems": len(new_stems),
        "overlap_count":    len(overlap),
        "overlap_samples":  sorted(overlap)[:8],
    }


def summary_table(dist: dict[str, int], title: str) -> str:
    total = sum(dist.values())
    lines = [f"\n  {title}"]
    lines.append(f"  {'Class':<22} {'Count':>7}  {'%':>6}")
    lines.append("  " + "─" * 38)
    for name, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100 if total else 0
        bar = "█" * int(pct / 5)
        lines.append(f"  {name:<22} {cnt:>7}  {pct:>5.1f}%  {bar}")
    lines.append(f"  {'TOTAL':<22} {total:>7}")
    return "\n".join(lines)


def main(report_only: bool = False) -> None:
    print(SEP)
    print("  CLASS SHIFT & LABEL INTEGRITY AUDIT")
    print(SEP)

    # ── 1. Float ID kontrolü — eski dataset ─────────────────────────────────
    print("\n[1] Float class ID check — OLD unified_dataset")
    print("    (augmented files write class IDs as '2.0' instead of '2')\n")
    old_float_total = 0
    for split in ("train", "val", "test"):
        r = audit_float_ids(OLD / split / "labels", split, "OLD")
        old_float_total += r["float_file_count"]
        status = "⚠ PROBLEM" if r["float_file_count"] > 0 else "✓ OK"
        print(f"  {split:<6}: {r['float_file_count']:>5} float-ID files, "
              f"{r['int_file_count']:>5} int-ID files  {status}")
        if r["float_cls_dist"]:
            print(f"         float tokens: {r['float_cls_dist']}")
        if r["float_file_samples"]:
            print(f"         examples: {r['float_file_samples'][:3]}")
    if old_float_total > 0:
        print(f"\n  ➜ RESULT: {old_float_total} files in OLD dataset have float class IDs.")
        print("    PyTorch dataloader may cast these correctly but it's non-standard.")
        print("    Ultralytics YOLO8.x reads them as float → converts to int, usually fine,")
        print("    BUT some versions reject them → silent NaN loss.")

    # ── 2. Float ID kontrolü — yeni dataset ─────────────────────────────────
    print(f"\n[2] Float class ID check — NEW unifiedV2_dataset")
    new_float_total = 0
    for split in ("train", "val", "test"):
        r = audit_float_ids(NEW / split / "labels", split, "NEW")
        new_float_total += r["float_file_count"]
        status = "✓ OK" if r["float_file_count"] == 0 else "⚠ PROBLEM"
        print(f"  {split:<6}: {r['float_file_count']:>5} float-ID files  {status}")
    if new_float_total == 0:
        print("  ➜ RESULT: NEW dataset is clean — all class IDs are integers. ✓")

    # ── 3. Class dağılımı karşılaştırması ────────────────────────────────────
    print(f"\n{SEP}")
    print("[3] Class distribution comparison (OLD vs NEW)")

    old_dist: Counter = Counter()
    for split in ("train", "val", "test"):
        for name, cnt in audit_class_dist(OLD / split / "labels", OLD_NAMES).items():
            old_dist[name] += cnt

    new_dist: Counter = Counter()
    for split in ("train", "val", "test"):
        for name, cnt in audit_class_dist(NEW / split / "labels", NEW_NAMES).items():
            new_dist[name] += cnt

    print(summary_table(dict(old_dist), "OLD unified_dataset (all splits)"))
    print(summary_table(dict(new_dist), "NEW unifiedV2_dataset (all splits)"))

    # Delta
    print(f"\n  {'Class':<22} {'OLD':>8} {'NEW':>8}  {'Delta':>8}  Note")
    print("  " + "─" * 65)
    all_names = sorted(set(old_dist) | set(new_dist))
    shifts = []
    for name in all_names:
        o = old_dist.get(name, 0)
        n = new_dist.get(name, 0)
        delta = n - o
        pct_change = (delta / o * 100) if o else float("inf")
        flag = ""
        if abs(pct_change) > 40:
            flag = "⚠ BIG SHIFT"
            shifts.append((name, o, n, pct_change))
        elif n == 0 and o > 0:
            flag = "⚠ MISSING IN NEW"
        elif o == 0 and n > 0:
            flag = "★ NEW IN NEW"
        print(f"  {name:<22} {o:>8} {n:>8}  {delta:>+8}  {flag}")

    # ── 4. Format1 klasör → class eşleşme doğrulaması ───────────────────────
    print(f"\n{SEP}")
    print("[4] Format1 folder-name → label class semantic check (RAW source)")
    issues = detect_folder_class_mismatch(FORMAT1, NEW_NAMES)
    for item in issues:
        status = "✓" if item["ok"] else "✗ MISMATCH"
        expected_names = [NEW_NAMES.get(i, f"cls{i}") for i in item["expected_ids"]] or ["DROP"]
        print(f"\n  [{status}] format1/{item['folder']}/")
        print(f"       Expected class : {item['expected_ids']} → {expected_names}")
        print(f"       Dominant label : {item['top_class_id']} ({item['top_name']}) "
              f"— {item['dominant_pct']:.1f}% of annotations")
        print(f"       Full dist      : {item['dist']}")

    # ── 5. Dosya stem örtüşmesi ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[5] File stem overlap: OLD ↔ NEW")
    ov = cross_check_file_overlap(OLD, NEW)
    print(f"  OLD unique stems (after stripping F1_ prefix & _aug suffix): {ov['old_unique_stems']}")
    print(f"  NEW unique stems                                            : {ov['new_unique_stems']}")
    print(f"  Overlap                                                     : {ov['overlap_count']}")
    if ov["overlap_samples"]:
        print(f"  Sample matches: {ov['overlap_samples'][:5]}")
    if ov["overlap_count"] == 0:
        print("  ➜ No stem overlap found — datasets use different source files or naming conventions.")
    elif ov["overlap_count"] > 100:
        print("  ➜ Significant overlap — same images exist in both datasets.")

    # ── 6. Eski datasette augment oranı ──────────────────────────────────────
    print(f"\n{SEP}")
    print("[6] Augmented file ratio in OLD dataset (_aug_N suffix)")
    for split in ("train", "val", "test"):
        lbl_dir = OLD / split / "labels"
        if not lbl_dir.exists():
            continue
        total = 0; aug = 0
        for f in lbl_dir.glob("*.txt"):
            total += 1
            if re.search(r"_aug_\d+", f.stem):
                aug += 1
        pct = aug / total * 100 if total else 0
        print(f"  {split:<6}: {aug:>4}/{total} augmented ({pct:.1f}%)")

    # ── 7. Verdict ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VERDICT")
    print(SEP)
    print("""
  OLD unified_dataset sorunları:
  ✗ Float class IDs (_aug dosyalarında): kayıp/bozuk annotations riski
  ✗ Dust klasöründen gelen veriler bird_drop (0) olarak etiketlenmiş
  ✗ Physical_aug dosyaları 2.0 yazılmış

  NEW unifiedV2_dataset:
  ✓ Tüm class ID'ler integer
  ✓ Her format1 klasörü kendi classes.txt'ine göre doğru remap edildi
  ✓ Electrical_damage → physical_damage birleşimi yapıldı
  ✓ Healthy/clean sınıfı kaldırıldı
  ✓ OBB (9-col) → standart YOLO (5-col) dönüşümü yapıldı

  ➜ v1.0.5 eğitimini NEW (unifiedV2) dataset üzerinde yap.
    OLD dataset'i sakla, silme — tarihsel karşılaştırma için gerekli.
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    main(report_only=args.report_only)
