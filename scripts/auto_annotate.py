#!/usr/bin/env python3
"""Auto-Annotate captured frames using a YOLO model."""

import argparse
from pathlib import Path

from ultralytics import YOLO

from scripts.utils.logger import get_logger

logger = get_logger(__name__)


def auto_annotate(source_dir: Path, model_path: Path, conf_threshold: float) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        logger.error("source_directory_not_found", path=str(source_dir))
        return

    if not model_path.exists():
        logger.error("model_not_found", path=str(model_path))
        return

    logger.info("loading_model", path=str(model_path))
    model = YOLO(str(model_path))

    images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpeg"))
    if not images:
        logger.warning("no_images_found", path=str(source_dir))
        return

    logger.info("starting_auto_annotation", count=len(images))
    annotated_count = 0

    for img_path in images:
        results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)
        
        txt_path = img_path.with_suffix(".txt")
        # Even if nothing is found, we might want to create an empty text file or not.
        # YOLO format: class_id x_center y_center width height
        boxes_written = 0
        with open(txt_path, "w", encoding="utf-8") as f:
            for result in results:
                # result.boxes contains bounding boxes
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        # normalized format (xywhn)
                        x, y, w, h = box.xywhn[0].tolist()
                        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                        boxes_written += 1
                        
        if boxes_written >= 0: # count all processed images
            annotated_count += 1
            
    # Generate classes.txt for Label Studio
    classes_txt = source_dir / "classes.txt"
    with open(classes_txt, "w", encoding="utf-8") as f:
        # model.names is a dictionary like {0: 'class1', 1: 'class2'}
        # we need to ensure they are written in order from 0 to max_id
        max_id = max(model.names.keys()) if model.names else -1
        for i in range(max_id + 1):
            class_name = model.names.get(i, f"class_{i}")
            f.write(f"{class_name}\n")
    logger.info("generated_classes_txt", path=str(classes_txt))
    
    logger.info("auto_annotation_complete", annotated=annotated_count, total=len(images))


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto Annotator for Captured Frames")
    parser.add_argument("--source", type=Path, default=Path("data/raw_data/captured"), help="Directory containing captured images")
    parser.add_argument("--model", type=Path, default=Path("models/yolo11n.pt"), help="Path to the YOLO model to use for annotation")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    args = parser.parse_args()

    auto_annotate(args.source, args.model, args.conf)


if __name__ == "__main__":
    main()
