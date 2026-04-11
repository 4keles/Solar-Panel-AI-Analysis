"""Minimal VOC XML to YOLO label converter skeleton."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def convert_single_voc_xml(xml_path: Path, class_to_idx: dict[str, int]) -> tuple[tuple[int, float, float, float, float], ...]:
    """Convert VOC objects to normalized YOLO tuples."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    if size is None:
        return tuple()
    width = float(size.findtext('width', default='0'))
    height = float(size.findtext('height', default='0'))
    if width <= 0 or height <= 0:
        return tuple()

    rows: list[tuple[int, float, float, float, float]] = []
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        if name not in class_to_idx:
            continue
        box = obj.find('bndbox')
        if box is None:
            continue
        xmin = float(box.findtext('xmin', default='0'))
        ymin = float(box.findtext('ymin', default='0'))
        xmax = float(box.findtext('xmax', default='0'))
        ymax = float(box.findtext('ymax', default='0'))

        cx = ((xmin + xmax) / 2.0) / width
        cy = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        rows.append((class_to_idx[name], cx, cy, bw, bh))

    return tuple(rows)


def write_yolo_label(label_path: Path, rows: tuple[tuple[int, float, float, float, float], ...]) -> None:
    """Write converted YOLO rows to txt file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open('w', encoding='utf-8') as f:
        for cls, cx, cy, bw, bh in rows:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
