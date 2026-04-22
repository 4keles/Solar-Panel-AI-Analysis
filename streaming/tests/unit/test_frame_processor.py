import pytest
import numpy as np
import time
import queue
from unittest.mock import MagicMock

from src.core.frame_processor import FrameProcessor, ProcessResult
from src.core.exceptions import ProcessorNotInitializedError

@pytest.fixture
def mock_yolo_model():
    model = MagicMock()
    model.names = {0: "physical_damage", 1: "dust_particle"}
    
    mock_result = MagicMock()
    mock_boxes = MagicMock()
    # Mocking tensors for CPU processing
    mock_boxes.xyxy.cpu().numpy.return_value = np.array([[10, 10, 50, 50], [20, 20, 40, 40]])
    mock_boxes.conf.cpu().numpy.return_value = np.array([0.95, 0.85])
    mock_boxes.cls.cpu().numpy.return_value = np.array([0, 1])
    mock_boxes.__len__.return_value = 2
    mock_result.boxes = mock_boxes
    
    model.predict.return_value = [mock_result]
    return model

@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_process_returns_original_frame_unchanged(mock_yolo_model, dummy_frame):
    in_q = queue.Queue()
    out_q = queue.Queue()
    proc = FrameProcessor(mock_yolo_model, in_q, out_q)
    
    original_sum = dummy_frame.sum()
    result = proc.process(dummy_frame)
    
    assert result.frame is dummy_frame
    assert result.frame.sum() == original_sum

def test_empty_frame_returns_no_detections(mock_yolo_model):
    in_q = queue.Queue()
    out_q = queue.Queue()
    proc = FrameProcessor(mock_yolo_model, in_q, out_q)
    
    empty_frame = np.zeros(0)
    result = proc.process(empty_frame)
    
    assert len(result.detections) == 0
    assert result.inference_ms == 0.0

def test_detections_have_correct_fields(mock_yolo_model, dummy_frame):
    in_q = queue.Queue()
    out_q = queue.Queue()
    proc = FrameProcessor(mock_yolo_model, in_q, out_q)
    
    result = proc.process(dummy_frame)
    
    assert len(result.detections) == 2
    
    det1 = result.detections[0]
    assert det1.class_id == 0
    assert det1.class_name == "physical_damage"
    assert det1.confidence == 0.95
    assert det1.bbox_xyxy == (10.0, 10.0, 50.0, 50.0)

def test_inference_ms_positive(mock_yolo_model, dummy_frame):
    # Simulating standard sleep to make time positive
    def mock_predict(*args, **kwargs):
        time.sleep(0.01)
        mock_result = MagicMock()
        mock_result.boxes = None
        return [mock_result]
        
    mock_yolo_model.predict.side_effect = mock_predict
    
    in_q = queue.Queue()
    out_q = queue.Queue()
    proc = FrameProcessor(mock_yolo_model, in_q, out_q)
    
    result = proc.process(dummy_frame)
    assert result.inference_ms > 0

def test_uninitialized_raises(dummy_frame):
    in_q = queue.Queue()
    out_q = queue.Queue()
    proc = FrameProcessor(None, in_q, out_q)
    
    with pytest.raises(ProcessorNotInitializedError):
        proc.start()
        
    with pytest.raises(ProcessorNotInitializedError):
        proc.process(dummy_frame)
