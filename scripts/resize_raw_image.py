#!/usr/bin/env python3
"""
resize_to_640_letterbox.py

Kullanım:
    python resize_to_640_letterbox.py /girdi/klasoru /cikti/klasoru --color 0 0 0 --recursive --overwrite

Requirements:
    pip install opencv-python

Açıklama:
    - Oran korunur (aspect-ratio), görüntü ortalanır ve kalan kısım pad color ile doldurulur.
    - BBox dönüşümü istemediğinizi söylediniz; bu script sadece görüntüleri dönüştürür.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

TARGET = 640
SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def letterbox(img: np.ndarray, target=TARGET, color=(0,0,0)):
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas = np.full((target, target, 3), color, dtype=np.uint8)
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return canvas

def process_dir(input_dir: Path, output_dir: Path, color=(0,0,0), recursive=False, overwrite=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(input_dir.rglob('*') if recursive else input_dir.iterdir())
    for p in files:
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXT:
            continue
        out_path = output_dir / p.name
        if out_path.exists() and not overwrite:
            print(f"Skip (exists): {out_path}")
            continue
        try:
            img = cv2.imread(str(p))
            if img is None:
                print(f"Cannot read: {p}")
                continue
            out = letterbox(img, target=TARGET, color=color)
            # JPEG quality default; use PNG if original was PNG to preserve alpha if any (alpha not handled here)
            ext = p.suffix.lower()
            if ext in ('.jpg', '.jpeg'):
                cv2.imwrite(str(out_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                cv2.imwrite(str(out_path), out)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Error processing {p}: {e}")

def parse_color(args):
    if args.color is None:
        return (0,0,0)
    if len(args.color) == 1:
        v = int(args.color[0])
        return (v, v, v)
    return tuple(int(x) for x in args.color[:3])

def main():
    ap = argparse.ArgumentParser(description="Letterbox images to 640x640 for labeling")
    ap.add_argument("input_dir", type=Path, help="Input folder with images")
    ap.add_argument("output_dir", type=Path, help="Output folder")
    ap.add_argument("--color", nargs='+', help="Pad color: one value (gray) or three values R G B, e.g. --color 0 0 0", default=None)
    ap.add_argument("--recursive", action="store_true", help="Process subfolders")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    color = parse_color(args)
    process_dir(args.input_dir, args.output_dir, color=color, recursive=args.recursive, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
