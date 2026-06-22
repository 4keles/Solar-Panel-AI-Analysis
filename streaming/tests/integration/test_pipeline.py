import pytest
import numpy as np
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2

@pytest.fixture
def mock_yolo_model():
    model = MagicMock()
    model.names = {0: "physical_damage", 1: "bird_drop"}
    
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu().numpy.return_value = np.array([[10, 10, 50, 50]])
    mock_boxes.conf.cpu().numpy.return_value = np.array([0.95])
    mock_boxes.cls.cpu().numpy.return_value = np.array([0])
    mock_boxes.__len__.return_value = 1
    mock_result.boxes = mock_boxes
    
    model.predict.return_value = [mock_result]
    return model

@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Sahte 10 karelik test videosu."""
    path = tmp_path / "test.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for _ in range(10):
        # Create a frame with some content
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
        out.write(frame)
    out.release()
    return path

def test_full_pipeline_integration(tmp_path, sample_video, mock_yolo_model):
    from src.core.model_loader import ModelLoader
    from src.core.source_manager import VideoSource
    from src.core.frame_processor import FrameProcessor
    from src.core.annotator import Annotator
    from src.core.recorder import VideoRecorder
    from src.utils.class_colors import generate_class_colors
    import queue

    with patch("src.core.model_loader.YOLO") as MockYOLO:
        MockYOLO.return_value = mock_yolo_model

        loader = ModelLoader()
        
        # Create a fake best.pt
        model_path = tmp_path / "best.pt"
        model_path.write_bytes(b"temp")
        
        model = loader.load(model_path)
        
        source = VideoSource(str(sample_video))
        out_q = queue.Queue(maxsize=10)
        
        proc = FrameProcessor(model, source._queue, out_q, conf=0.25, iou=0.45, device="cpu")
        ann = Annotator(class_colors=generate_class_colors(model.names), conf_threshold=0.0)
        rec = VideoRecorder(output_dir=tmp_path / "recordings", fps=30, resolution=(640, 480))

        source.open()
        proc.start()
        rec.start("test_integration")
        
        processed = 0
        timeout_at = time.time() + 5.0
        
        # Read from output queue
        while True:
            if time.time() > timeout_at:
                break
                
            try:
                result = out_q.get(timeout=0.2)
                if result.frame is not None and result.frame.size > 0:
                    annotated = ann.draw(result.frame, result)
                    rec.write(annotated)
                    processed += 1
                else:
                    # Received EOF signal
                    break
            except queue.Empty:
                if not source._is_running:
                    break

        proc.stop()
        source.release()
        summary = rec.stop()

        # Depending on queue timing it might not process exactly 10 before EOF but should process some
        assert processed > 0
        assert summary.output_path.exists()
        assert summary.file_size_bytes > 0
