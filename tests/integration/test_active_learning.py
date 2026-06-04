import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from scripts.active_learning_pipeline import auto_annotate, augment_dataset, merge_dataset

def test_active_learning_flow(tmp_path: Path):
    """Test the entire active learning flow (Annotation -> Augmentation -> Merge)."""
    # 1. Setup mock captured data
    capture_dir = tmp_path / "captured"
    capture_dir.mkdir()
    
    # Create a mock image
    mock_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(mock_img, (100, 100), (200, 200), (255, 255, 255), -1)
    img_path = capture_dir / "test_capture.jpg"
    cv2.imwrite(str(img_path), mock_img)
    
    # 2. Mock auto-annotation (since we don't want to load a real YOLO model in tests)
    # We just create a .txt file directly
    txt_path = capture_dir / "test_capture.txt"
    txt_path.write_text("0 0.5 0.5 0.1 0.1\n")
    
    # 3. Augmentation
    aug_out = tmp_path / "augmented_temp"
    aug_out.mkdir()
    # We will use the common test config or a dummy config if we don't want to rely on the real one.
    # To keep it simple, we just mock the augmented files. 
    # Real augmentation test is in test_augment.py.
    # But let's verify merge_dataset works with nested directories.
    aug_images = aug_out / "images"
    aug_labels = aug_out / "labels"
    aug_images.mkdir()
    aug_labels.mkdir()
    
    cv2.imwrite(str(aug_images / "test_capture_aug_0.jpg"), mock_img)
    (aug_labels / "test_capture_aug_0.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    
    # 4. Merge Dataset
    target_base = tmp_path / "finetune"
    
    # Merge augmented
    merge_dataset(aug_out, target_base, split_ratio=1.0) # all to train
    
    # Verify merge augmented
    assert (target_base / "images" / "train" / "test_capture_aug_0.jpg").exists()
    assert (target_base / "labels" / "train" / "test_capture_aug_0.txt").exists()
    
    # Merge originals
    merge_dataset(capture_dir, target_base, split_ratio=1.0)
    
    # Verify merge originals
    assert (target_base / "images" / "train" / "test_capture.jpg").exists()
    assert (target_base / "labels" / "train" / "test_capture.txt").exists()
    
    # Verify archive
    archive_dirs = list((capture_dir / "archive").glob("*"))
    assert len(archive_dirs) == 1
    assert (archive_dirs[0] / "test_capture.jpg").exists()
    assert not (capture_dir / "test_capture.jpg").exists()
