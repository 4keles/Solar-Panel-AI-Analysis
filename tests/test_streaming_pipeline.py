import sys
from pathlib import Path
import numpy as np
import cv2

# Project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from streaming.src.core.exceptions import ConnectionStatus
from streaming.src.core.source_manager import ResilientVideoSource, SourceType
from streaming.src.core.frame_processor import ThermalConverter, FrameProcessor
from streaming.src.core.annotator import Annotator

def test_thermal_converter():
    print("Testing ThermalConverter...")
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 1. Grayscale
    conv = ThermalConverter("grayscale")
    res = conv.convert(frame)
    assert res.shape == (100, 100, 3), f"Expected shape (100, 100, 3), got {res.shape}"
    # Verify B, G, R channels are identical (grayscale converted BGR)
    assert np.allclose(res[:, :, 0], res[:, :, 1]), "B and G channels are not identical in grayscale"
    assert np.allclose(res[:, :, 1], res[:, :, 2]), "G and R channels are not identical in grayscale"
    print("Grayscale conversion OK!")

    # 2. Inferno
    conv_inferno = ThermalConverter("inferno")
    res_inferno = conv_inferno.convert(frame)
    assert res_inferno.shape == (100, 100, 3)
    print("Inferno conversion OK!")
    
    # 3. None/Invalid
    conv_invalid = ThermalConverter("invalid_mode")
    res_invalid = conv_invalid.convert(frame)
    assert res_invalid.shape == (100, 100, 3)
    print("Fallback conversion OK!")

def test_annotator_draw_hud():
    print("Testing Annotator OSD...")
    annotator = Annotator(class_colors={}, conf_threshold=0.25)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test connected status hud
    hud = annotator.draw_hud(
        frame,
        fps=30.0,
        source_label="http://192.168.1.50/video",
        recording=True,
        inference_ms=12.3,
        conn_status="connected",
        thermal_mode="inferno"
    )
    assert hud.shape == (480, 640, 3)
    print("OSD Drawing (Connected / Thermal) OK!")

    # Test reconnecting status hud
    hud_reconn = annotator.draw_hud(
        frame,
        fps=0.0,
        source_label="rtsp://192.168.0.50/live",
        recording=False,
        inference_ms=None,
        conn_status="reconnecting",
        thermal_mode="grayscale"
    )
    assert hud_reconn.shape == (480, 640, 3)
    print("OSD Drawing (Reconnecting / Grayscale) OK!")

def test_resilient_video_source():
    print("Testing ResilientVideoSource instantiation...")
    # Just instantiate and check properties
    source = ResilientVideoSource("0", max_queue_size=2, max_retries=5, retry_interval_sec=1.0)
    assert source._type == SourceType.CAMERA
    assert source.max_retries == 5
    assert source.retry_interval_sec == 1.0
    assert source._connection_status == ConnectionStatus.DISCONNECTED
    print("ResilientVideoSource properties OK!")

def test_ui_launch():
    print("Testing ControlPanel UI construction...")
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QTimer
        from streaming.src.ui.control_panel import ControlPanel

        # Mock config
        config = {
            "display": {
                "window_title": "Test Title",
                "show_window": False
            },
            "source": {
                "type": "camera",
                "camera_id": 0
            },
            "model": {
                "path": "models/v1.0.3/best.onnx",
                "conf": 0.25,
                "device": "cpu"
            },
            "recording": {
                "enabled": False,
                "video_output_dir": "output/recordings",
                "capture_output_dir": "data/raw_data/captured"
            }
        }

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Instantiate window - this runs _build_ui() and layout construction
        config_path = Path("streaming/configs/streaming.yaml")
        window = ControlPanel(config, config_path=config_path)
        
        # Verify layouts and widgets exist
        assert window.source_selector is not None
        assert window.thermal_control is not None
        assert window.path_manager is not None
        
        # Schedule exit
        QTimer.singleShot(100, app.quit)
        app.exec()
        print("ControlPanel UI construction OK!")
    except SystemExit:
        # Qt platform xcb could raise SystemExit if headless
        print("Skipping UI launch test (SystemExit: No display/X11 connection).")
    except Exception as e:
        if "xcb" in str(e) or "display" in str(e).lower() or "qt.qpa" in str(e).lower():
            print(f"Skipping UI launch test (no X11 DISPLAY available): {e}")
        else:
            raise e

if __name__ == "__main__":
    test_thermal_converter()
    test_annotator_draw_hud()
    test_resilient_video_source()
    test_ui_launch()
    print("\nAll pipeline components verified successfully!")
