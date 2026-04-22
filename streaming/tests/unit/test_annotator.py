import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.core.annotator import Annotator
from src.core.frame_processor import ProcessResult, Detection
from src.utils.class_colors import PRESET_COLORS

@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def dummy_result(dummy_frame):
    return ProcessResult(
        frame=dummy_frame,
        detections=[
            Detection(0, "physical_damage", 0.90, (50.0, 50.0, 200.0, 200.0)),
            Detection(1, "dust_particle", 0.10, (300.0, 300.0, 400.0, 400.0))  # Low conf
        ],
        inference_ms=10.0
    )

def test_draw_does_not_modify_original(dummy_frame, dummy_result):
    ann = Annotator(class_colors=PRESET_COLORS, conf_threshold=0.25)
    
    out = ann.draw(dummy_frame, dummy_result)
    
    # Must be a copy
    assert out is not dummy_frame
    # Original frame should still be all zeros
    assert dummy_frame.sum() == 0
    # Modifed frame should have drawn pixels
    assert out.sum() > 0

def test_draw_empty_detections(dummy_frame):
    ann = Annotator(class_colors=PRESET_COLORS, conf_threshold=0.25)
    
    empty_result = ProcessResult(frame=dummy_frame, detections=[], inference_ms=0)
    out = ann.draw(dummy_frame, empty_result)
    
    assert out.sum() == 0

def test_hud_shows_fps_and_source(dummy_frame):
    ann = Annotator(class_colors=PRESET_COLORS, conf_threshold=0.25)
    out = ann.draw_hud(dummy_frame, fps=30.0, source_label="Test")
    
    # Just asserting it drew something, we can't easily parse text from pixel without OCR
    assert out.sum() > 0

def test_recording_indicator_shown(dummy_frame):
    ann = Annotator(class_colors=PRESET_COLORS, conf_threshold=0.25)
    out_no_rec = ann.draw_hud(dummy_frame, fps=30.0, source_label="Test", recording=False)
    out_rec = ann.draw_hud(dummy_frame, fps=30.0, source_label="Test", recording=True)
    
    # Frame with recording indicator should have more non-zero pixels (or at least differ)
    assert out_rec.sum() != out_no_rec.sum()
