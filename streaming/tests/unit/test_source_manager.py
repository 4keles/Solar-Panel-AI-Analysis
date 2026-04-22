import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np

from src.core.source_manager import VideoSource, SourceType
from src.core.exceptions import SourceOpenError, SourceNotOpenError

def test_detect_camera_source():
    assert VideoSource(0)._type == SourceType.CAMERA
    assert VideoSource("1")._type == SourceType.CAMERA

def test_detect_video_source():
    assert VideoSource("test.mp4")._type == SourceType.VIDEO
    assert VideoSource(Path("video.avi"))._type == SourceType.VIDEO

def test_detect_rtsp_source():
    assert VideoSource("rtsp://ip:port/stream")._type == SourceType.RTSP

def test_read_without_open_raises():
    src = VideoSource("test.mp4")
    with pytest.raises(SourceNotOpenError):
        src.read()

@patch("src.core.source_manager.cv2.VideoCapture")
def test_open_missing_file_raises(mock_vc):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_vc.return_value = mock_cap

    src = VideoSource("missing.mp4")
    with pytest.raises(SourceOpenError):
        src.open()

@patch("src.core.source_manager.cv2.VideoCapture")
def test_context_manager_closes(mock_vc):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 100.0  # mock fps/dims
    mock_cap.read.return_value = (False, None)
    mock_vc.return_value = mock_cap

    with VideoSource("test.mp4") as src:
        assert src._is_running is True
        
    assert src._is_running is False
    mock_cap.release.assert_called_once()

@patch("src.core.source_manager.cv2.VideoCapture")
def test_is_file_and_is_live(mock_vc):
    src1 = VideoSource("test.mp4")
    assert src1.is_file() is True
    assert src1.is_live() is False

    src2 = VideoSource(0)
    assert src2.is_file() is False
    assert src2.is_live() is True

@patch("src.core.source_manager.cv2.VideoCapture")
def test_async_capture_logic(mock_vc):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0
    # Two valid frames then end
    mock_cap.read.side_effect = [(True, np.zeros((10,10,3))), (True, np.zeros((10,10,3))), (False, None)]
    mock_vc.return_value = mock_cap

    with VideoSource("test.mp4") as src:
        ret1, frame1 = src.read()
        assert ret1 is True
        assert frame1 is not None
        
        ret2, frame2 = src.read()
        assert ret2 is True
        
        ret3, frame3 = src.read()
        assert ret3 is False
