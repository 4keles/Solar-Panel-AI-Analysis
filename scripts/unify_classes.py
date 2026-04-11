#!/usr/bin/env python3
"""
Unify and merge multiple YOLO datasets into a central master dataset.
This script resolves class index mismatches across different downloaded projects
by translating local classes.txt indices to a global Master Class Mapping.
"""

import os
import shutil
import random
from pathlib import Path

GLOBAL_CLASSES = {
    0: "physical_damage",
    1: "bird_drop",
    2: "dust_particle",
    3: "leaf",
    4: "snow",
    5: "electrical_damage_hotspot",
    6: "healthy",
    7: "bird_feather"
}

# Mapping of potential variant names to global names
NAME_NORMALIZATION = {
    "dust_partical": "dust_particle",
    "electrical_damage": "electrical_damage_hotspot",
}

def get_inverse_global():
    return {v: k for k, v in GLOBAL_CLASSES.items()}

def parse_local_classes(classes_path: Path):
    if not classes_path.exists():
        return None
    lines = classes_path.read_text(encoding="utf-8").strip().splitlines()
    mapping = {}
    inv_global = get_inverse_global()
    for local_idx, line in enumerate(lines):
        name = line.strip()
        if not name:
            continue
        # Normalize
        norm_name = NAME_NORMALIZATION.get(name, name)
        if norm_name in inv_global:
            mapping[local_idx] = inv_global[norm_name]
        else:
            print(f"Warning: Unknown class '{name}' in {classes_path}")
    return mapping

def convert_label_file(src_label: Path, dst_label: Path, mapping: dict):
    if not src_label.exists():
        return
    lines = src_label.read_text("utf-8").strip().splitlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        local_idx = int(parts[0])
        if local_idx in mapping:
            global_idx = mapping[local_idx]
            parts[0] = str(global_idx)
            new_lines.append(" ".join(parts))
        else:
            # Drop unknown classes
            pass
    if new_lines:
        dst_label.write_text("\n".join(new_lines) + "\n", "utf-8")

def process_modality(modality="rgb", split_ratio=0.8):
    base_dir = Path(f"/home/kayra/git/solar_panel_od/data/processed_data/{modality}")
    master_dir = Path(f"/home/kayra/git/solar_panel_od/data/processed_data/{modality}_master")
    
    if not base_dir.exists():
        print(f"Modality {modality} not found. Skipping.")
        return

    # Delete existing master
    if master_dir.exists():
        shutil.rmtree(master_dir)
        
    for split in ["train", "val"]:
        (master_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (master_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Make master Data yaml
    yaml_content = f"path: {master_dir.resolve()}\n"
    yaml_content += "train: train/images\n"
    yaml_content += "val: val/images\n"
    yaml_content += f"nc: {len(GLOBAL_CLASSES)}\nnames:\n"
    for k, v in sorted(GLOBAL_CLASSES.items()):
        yaml_content += f"  {k}: {v}\n"
    (master_dir / "data.yaml").write_text(yaml_content)

    for item in base_dir.iterdir():
        if not item.is_dir() or item.name.endswith("_master"):
            continue
            
        print(f"Processing dataset chunk: {item.name}")
        local_classes_mapping = None
        
        # Determine class mapping
        if (item / "classes.txt").exists():
            local_classes_mapping = parse_local_classes(item / "classes.txt")
        elif item.name == "mvp_v1" or item.name == "mvp_test_v1":
            # Hardcoded mvp mapping
            local_classes_mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}
        else:
            # Try to find classes.txt deeper
            deep_classes = list(item.rglob("classes.txt"))
            if deep_classes:
                local_classes_mapping = parse_local_classes(deep_classes[0])
            else:
                print(f"  Warning: No classes.txt found for {item.name}, skipping.")
                continue
                
        # Find all images
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            images.extend(item.rglob(ext))
            
        print(f"  Found {len(images)} images.")
        
        # Decide splits
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        def copy_files(img_list, split):
            for img_path in img_list:
                # Find matching label
                # In YOLO, it usually shares the same stem
                label_candidates = list(img_path.parent.glob(f"{img_path.stem}.txt"))
                # Sometimes labels are in a siblings folder
                if not label_candidates and "images" in str(img_path):
                    alt_label_dir = Path(str(img_path.parent).replace("images", "labels"))
                    label_candidates = list(alt_label_dir.glob(f"{img_path.stem}.txt"))
                
                label_path = label_candidates[0] if label_candidates else None
                
                # Copy Image
                dst_img = master_dir / split / "images" / f"{item.name}_{img_path.name}"
                shutil.copy2(img_path, dst_img)
                
                if label_path and label_path.exists():
                    dst_label = master_dir / split / "labels" / f"{item.name}_{label_path.name}"
                    convert_label_file(label_path, dst_label, local_classes_mapping)

        copy_files(train_imgs, "train")
        copy_files(val_imgs, "val")

if __name__ == "__main__":
    random.seed(42)
    process_modality("rgb")
    process_modality("thermal")
    print("Done unifying classes and generating master datasets.")
