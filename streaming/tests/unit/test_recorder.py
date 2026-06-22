import pytest
import numpy as np
from pathlib import Path

from src.core.recorder import VideoRecorder
from src.core.exceptions import RecorderAlreadyRunningError, RecorderNotRunningError

@pytest.fixture
def temp_output_dir(tmp_path):
    out = tmp_path / "recordings"
    return out

@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_start_creates_file(temp_output_dir):
    rec = VideoRecorder(output_dir=temp_output_dir, fps=30.0, resolution=(640, 480))
    file_path = rec.start("test_video.mp4")
    
    assert file_path.exists()
    assert rec.is_recording() is True
    
    rec.stop()

def test_write_increments_frame_count(temp_output_dir, dummy_frame):
    rec = VideoRecorder(output_dir=temp_output_dir, fps=30.0, resolution=(640, 480))
    rec.start()
    
    for _ in range(5):
        rec.write(dummy_frame)
        
    summary = rec.stop()
    assert summary.frame_count == 5

def test_stop_returns_summary(temp_output_dir, dummy_frame):
    rec = VideoRecorder(output_dir=temp_output_dir, fps=30.0, resolution=(640, 480))
    rec.start("summary_test.mp4")
    
    for _ in range(30):
        rec.write(dummy_frame)
        
    summary = rec.stop()
    assert summary.output_path.name == "summary_test.mp4"
    assert summary.frame_count == 30
    assert summary.duration_sec == 1.0  # 30 frames / 30 fps
    assert summary.file_size_bytes > 0

def test_double_start_raises(temp_output_dir):
    rec = VideoRecorder(output_dir=temp_output_dir, fps=30.0, resolution=(640, 480))
    rec.start()
    
    with pytest.raises(RecorderAlreadyRunningError):
        rec.start()
        
    rec.stop()

def test_stop_without_start_raises(temp_output_dir):
    rec = VideoRecorder(output_dir=temp_output_dir, fps=30.0, resolution=(640, 480))
    
    with pytest.raises(RecorderNotRunningError):
        rec.stop()

def test_output_only_in_recordings_dir(temp_output_dir):
    rec = VideoRecorder(output_dir=temp_output_dir, fps=30.0, resolution=(640, 480))
    file_path = rec.start()
    
    # Check parent dir is exactly temp_output_dir
    assert file_path.parent == temp_output_dir
    rec.stop()
