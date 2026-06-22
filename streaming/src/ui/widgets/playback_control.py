"""PlaybackControlWidget — Video playback controls and timeline slider."""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox
from PyQt6.QtCore import Qt, pyqtSignal

class PlaybackControlWidget(QWidget):
    """Widget containing play/pause button, timeline slider, and time label."""
    
    play_toggled = pyqtSignal(bool)   # True if playing, False if paused
    seek_requested = pyqtSignal(int)  # Emits frame index to seek to
    speed_changed = pyqtSignal(float) # Emits playback speed multiplier
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("playbackControlWidget")
        self.setStyleSheet("background-color: #1E2A3A; border-radius: 4px; margin-top: 4px;")
        
        self.is_playing = True
        self._total_frames = 0
        self._current_frame = 0
        self._fps = 30.0
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(12)
        
        # Play/Pause Button
        self.btn_play_pause = QPushButton("⏸")
        self.btn_play_pause.setFixedSize(30, 30)
        self.btn_play_pause.setStyleSheet("""
            QPushButton { background-color: #3B82F6; color: white; border: none; border-radius: 15px; font-weight: bold; }
            QPushButton:hover { background-color: #2563EB; }
        """)
        self.btn_play_pause.clicked.connect(self._toggle_play)
        layout.addWidget(self.btn_play_pause)
        
        # Timeline Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #161B27; height: 6px; background: #0F172A; border-radius: 3px; }
            QSlider::handle:horizontal { background: #38BDF8; width: 14px; margin: -4px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #3B82F6; border-radius: 3px; }
        """)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        layout.addWidget(self.slider)
        
        # Time Label
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("color: #94A3B8; font-size: 11px; font-family: monospace;")
        self.lbl_time.setMinimumWidth(80)
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_time)
        
        # Speed ComboBox
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.5x", "1.0x", "1.5x", "2.0x", "4.0x"])
        self.combo_speed.setCurrentText("1.0x")
        self.combo_speed.setStyleSheet("""
            QComboBox { background-color: #0F172A; color: #38BDF8; border: 1px solid #1E2A3A; border-radius: 3px; font-size: 11px; padding: 2px 4px; }
            QComboBox::drop-down { border: none; }
        """)
        self.combo_speed.currentTextChanged.connect(self._on_speed_changed)
        layout.addWidget(self.combo_speed)
        
        self._is_dragging = False

    def setup(self, total_frames: int, fps: float, is_image_dir: bool = False):
        self._is_image_dir = is_image_dir
        self._total_frames = total_frames
        self._fps = fps if fps > 0 else 30.0
        self.slider.setRange(0, max(0, total_frames - 1))
        self.update_position(0)
        self.is_playing = True
        self.btn_play_pause.setText("⏸")
        
    def _toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play_pause.setText("⏸" if self.is_playing else "▶")
        self.play_toggled.emit(self.is_playing)
        
    def update_position(self, current_frame: int):
        if self._is_dragging:
            return
        self._current_frame = min(current_frame, self._total_frames)
        self.slider.blockSignals(True)
        self.slider.setValue(self._current_frame)
        self.slider.blockSignals(False)
        self._update_time_label()

    def _on_slider_pressed(self):
        self._is_dragging = True
        
    def _on_slider_moved(self, position: int):
        self._current_frame = position
        self._update_time_label()
        
    def _on_slider_released(self):
        self._is_dragging = False
        self.seek_requested.emit(self.slider.value())
        
    def _update_time_label(self):
        if getattr(self, '_is_image_dir', False):
            self.lbl_time.setText(f"Resim {self._current_frame + 1} / {self._total_frames}")
        else:
            curr_sec = int(self._current_frame / self._fps)
            tot_sec = int(self._total_frames / self._fps)
            self.lbl_time.setText(f"{curr_sec//60:02d}:{curr_sec%60:02d} / {tot_sec//60:02d}:{tot_sec%60:02d}")

    def _on_speed_changed(self, text: str):
        try:
            speed = float(text.replace("x", ""))
            self.speed_changed.emit(speed)
        except ValueError:
            pass
