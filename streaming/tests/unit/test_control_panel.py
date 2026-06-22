import pytest
import tkinter as tk
from unittest.mock import patch, MagicMock

from src.ui.control_panel import ControlPanel

@pytest.fixture
def mock_tkinter(monkeypatch):
    class MockTk:
        def __init__(self):
            self.title = MagicMock()
            self.protocol = MagicMock()
            self.after = MagicMock()
            self.update = MagicMock()
            self.destroy = MagicMock()

    class MockStringVar:
        def __init__(self, value=""):
            self.value = value
        def get(self): return self.value
        def set(self, val): self.value = val

    class MockDoubleVar(MockStringVar): pass
    
    monkeypatch.setattr(tk, "Tk", MockTk)
    monkeypatch.setattr(tk, "StringVar", MockStringVar)
    monkeypatch.setattr(tk, "DoubleVar", MockDoubleVar)
    monkeypatch.setattr(tk, "Canvas", MagicMock())
    return MockTk

@patch("src.ui.control_panel.ttk")
def test_control_panel_init(mock_ttk, mock_tkinter):
    root = tk.Tk()
    config = {"display": {"window_title": "Test"}}
    
    panel = ControlPanel(root, config)
    root.title.assert_called_with("Test")
    assert panel.is_running is False
    
@patch("src.ui.control_panel.ttk")
def test_toggle_recording(mock_ttk, mock_tkinter):
    root = tk.Tk()
    panel = ControlPanel(root, {})
    
    panel._recorder = MagicMock()
    panel._recorder.is_recording.return_value = False
    
    panel.btn_rec = MagicMock()
    
    # Enable recording
    panel.toggle_record()
    panel._recorder.start.assert_called_once()
    
    # Disable recording
    panel._recorder.is_recording.return_value = True
    panel.toggle_record()
    panel._recorder.stop.assert_called_once()
