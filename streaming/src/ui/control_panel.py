"""ControlPanel — Qt6 Main Window for Solar Panel OD Streaming.

Replaces the legacy Tkinter ControlPanel with a modern PyQt6 interface.
Core pipeline components (VideoSource, FrameProcessor, Annotator, Recorder)
are unchanged — only the UI layer is swapped.
"""

import queue
import sys
import time
import cv2
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton,
    QSlider, QGroupBox, QSizePolicy, QStatusBar,
    QFrame, QSpacerItem, QScrollArea, QSplitter,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QIcon

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
    _DEVICE = "cuda:0" if _CUDA_AVAILABLE else "cpu"
    _DEVICE_LABEL = f"GPU: {torch.cuda.get_device_name(0)}" if _CUDA_AVAILABLE else "CPU (CUDA yok)"
except Exception:
    _CUDA_AVAILABLE = False
    _DEVICE = "cpu"
    _DEVICE_LABEL = "CPU"

_scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from utils.logger import get_logger

from ..core.model_loader import ModelLoader
from ..core.source_manager import ResilientVideoSource, ConnectionStatus, ImageDirectorySource
from ..core.frame_processor import FrameProcessor
from ..core.annotator import Annotator
from ..core.recorder import VideoRecorder
from ..core.exceptions import SourceOpenError
from ..utils.class_colors import generate_class_colors
from ..utils.fps_counter import FPSCounter
from .widgets import (
    VideoWidget, StatsPanel, SourceSelectorWidget,
    ThermalControlWidget, PathManagerBar, HelpWindow,
    PlaybackControlWidget, ImageGalleryWidget
)

logger = get_logger(__name__)

# ── Model Type Badge Colors ────────────────────────────────────────────────
_MODEL_TYPE_STYLES = {
    ".pt":      ("PT",     "#FBBF24", "#1A1200"),
    ".onnx":    ("ONNX",   "#34D399", "#001A0D"),
    ".engine":  ("TRT",    "#A78BFA", "#0D001A"),
}


class _PipelineLoader(QThread):
    """Loads the model in a background thread so the UI doesn't freeze."""
    loaded = pyqtSignal(object, dict)   # model, class_names
    failed = pyqtSignal(str)

    def __init__(self, loader: ModelLoader, model_path: Path):
        super().__init__()
        self._loader = loader
        self._model_path = model_path

    def run(self):
        try:
            model = self._loader.load(self._model_path)
            class_names = self._loader.get_class_names()
            self.loaded.emit(model, class_names)
        except Exception as e:
            self.failed.emit(str(e))


class ControlPanel(QMainWindow):
    """Qt6 main window — orchestrates the streaming pipeline."""

    def __init__(self, config: dict, config_path: Path | str = "configs/streaming.yaml", parent=None):
        super().__init__(parent)
        self.config = config
        self.config_path = Path(config_path)

        # ── Pipeline State ─────────────────────────────────────
        self.model_loader = ModelLoader()
        self._source: ResilientVideoSource | None = None
        self._processor: FrameProcessor | None = None
        self._annotator: Annotator | None = None
        self._recorder: VideoRecorder | None = None
        self._loader_thread: _PipelineLoader | None = None

        self.q_out: queue.Queue = queue.Queue(maxsize=3)
        self.is_running = False
        self._fps_counter = FPSCounter()
        self._available_models: list[Path] = []


        # ── Window Setup ────────────────────────────────────────
        title = config.get("display", {}).get("window_title", "Solar Panel OD — Streaming")
        self.setWindowTitle(title)
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        # ── Build UI ────────────────────────────────────────────
        self._build_ui()
        self._populate_models()
        self._apply_initial_conf()

        # ── Update Timer (15 ms → ~66 fps ceiling) ─────────────
        self._timer = QTimer(self)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._update_frame)
        self._timer.start()

    # ═══════════════════════════════════════════════════════════
    #  UI Construction
    # ═══════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("mainSplitter")
        splitter.setStyleSheet("""
            QSplitter#mainSplitter::handle {
                background-color: #1E2A3A;
                width: 3px;
            }
            QSplitter#mainSplitter::handle:hover {
                background-color: #3B82F6;
            }
        """)

        # Left sidebar
        sidebar = self._build_sidebar()
        sidebar.setMaximumWidth(450)
        splitter.addWidget(sidebar)

        # Right: vertical splitter for video and stats panel
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setObjectName("rightSplitter")
        right_splitter.setStyleSheet("""
            QSplitter#rightSplitter::handle {
                background-color: #1E2A3A;
                height: 3px;
            }
            QSplitter#rightSplitter::handle:hover {
                background-color: #3B82F6;
            }
        """)

        # Container for video, gallery and playback controls
        video_gallery_widget = QWidget()
        vg_layout = QHBoxLayout(video_gallery_widget)
        vg_layout.setContentsMargins(0, 0, 0, 0)
        vg_layout.setSpacing(0)

        video_container = QWidget()
        v_layout = QVBoxLayout(video_container)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(0)
        
        self.video_widget = VideoWidget()
        self.video_widget.prev_clicked.connect(self._on_prev_image)
        self.video_widget.next_clicked.connect(self._on_next_image)
        v_layout.addWidget(self.video_widget, stretch=1)
        
        self.playback_control = PlaybackControlWidget()
        self.playback_control.hide()  # Hidden by default
        self.playback_control.play_toggled.connect(self._on_play_toggled)
        self.playback_control.seek_requested.connect(self._on_seek_requested)
        self.playback_control.speed_changed.connect(self._on_playback_speed_changed)
        v_layout.addWidget(self.playback_control)
        
        vg_layout.addWidget(video_container, stretch=1)
        
        self.image_gallery = ImageGalleryWidget()
        self.image_gallery.hide()
        self.image_gallery.image_selected.connect(self._on_gallery_image_selected)
        vg_layout.addWidget(self.image_gallery)
        
        right_splitter.addWidget(video_gallery_widget)

        self.stats_panel = StatsPanel()
        right_splitter.addWidget(self.stats_panel)

        right_splitter.setSizes([600, 100])
        right_splitter.setCollapsible(0, False)
        right_splitter.setCollapsible(1, False)

        splitter.addWidget(right_splitter)
        
        # Set initial sizes: sidebar = 290px, rest is flexible
        splitter.setSizes([290, 990])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        main_layout.addWidget(splitter)

        # Status Bar
        self._status_bar = QStatusBar()
        self._status_bar.setFixedHeight(26)
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready  —  Select a model and source, then click Start.")

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(260)

        # Main layout holds the scroll area
        main_layout = QVBoxLayout(sidebar)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scroll area setup
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #161B27;
                width: 6px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #1E2A3A;
                min-height: 20px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #3B82F6;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
        """)

        # Container widget for scroll content
        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        scroll_content.setStyleSheet("QWidget#scrollContent { background-color: #161B27; }")

        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(12)

        # ── App Title ──────────────────────────────────────────
        title_lbl = QLabel("Solar Panel OD")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        title_lbl.setFont(font)
        title_lbl.setStyleSheet("color: #F1F5F9; padding-bottom: 4px;")
        layout.addWidget(title_lbl)

        sub_lbl = QLabel("Real-Time Object Detection")
        sub_lbl.setStyleSheet("color: #475569; font-size: 11px; padding-bottom: 8px;")
        layout.addWidget(sub_lbl)

        # Divider
        layout.addWidget(self._divider())

        # ── Device Badge ───────────────────────────────────────
        obj_name = "deviceBadge" if _CUDA_AVAILABLE else "deviceBadgeCPU"
        self._device_badge = QLabel(_DEVICE_LABEL)
        self._device_badge.setObjectName(obj_name)
        self._device_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._device_badge.setWordWrap(True)
        layout.addWidget(self._device_badge)

        # ── Source Group (New SourceSelectorWidget) ────────────
        self.source_selector = SourceSelectorWidget()
        src_type = self.config.get("source", {}).get("type", "camera")
        if src_type == "camera":
            src_val = str(self.config.get("source", {}).get("camera_id", 0))
        elif src_type == "video":
            src_val = self.config.get("source", {}).get("video_path", "")
        elif src_type == "ip":
            src_val = self.config.get("source", {}).get("ip_url", "")
        elif src_type == "rtsp":
            src_val = self.config.get("source", {}).get("rtsp_url", "")
        elif src_type == "rtmp":
            src_val = self.config.get("source", {}).get("rtmp_url", "")
        else:
            src_val = "0"
        self.source_selector.set_source(src_type, src_val)
        layout.addWidget(self.source_selector)

        # ── Model Group ────────────────────────────────────────
        mdl_grp = QGroupBox("MODEL")
        mdl_layout = QVBoxLayout(mdl_grp)
        mdl_layout.setSpacing(6)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(240)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        mdl_layout.addWidget(self.model_combo)

        # Model type badge row
        badge_row = QHBoxLayout()
        badge_row.setSpacing(6)
        for ext, (label, fg, bg) in _MODEL_TYPE_STYLES.items():
            b = QLabel(label)
            b.setStyleSheet(
                f"background-color:{bg}; color:{fg}; border:1px solid {fg}40;"
                f"border-radius:3px; padding:1px 5px; font-size:10px; font-weight:700;"
            )
            badge_row.addWidget(b)
        badge_row.addStretch()
        mdl_layout.addLayout(badge_row)
        layout.addWidget(mdl_grp)

        # ── Thermal Control Group ──────────────────────────────
        self.thermal_control = ThermalControlWidget()
        self.thermal_control.mode_changed.connect(self._on_thermal_mode_changed)
        self.thermal_control.intensity_changed.connect(self._on_thermal_intensity_changed)
        layout.addWidget(self.thermal_control)

        # ── Confidence Group ───────────────────────────────────
        conf_grp = QGroupBox("CONFIDENCE THRESHOLD")
        conf_layout = QVBoxLayout(conf_grp)
        conf_layout.setSpacing(6)

        # Tespit (Model)
        conf_top = QHBoxLayout()
        conf_top.addWidget(QLabel("Detection (Model):"))
        conf_top.addStretch()
        self._conf_value_lbl = QLabel()
        self._conf_value_lbl.setStyleSheet("color: #38BDF8; font-weight: 700;")
        conf_top.addWidget(self._conf_value_lbl)
        conf_layout.addLayout(conf_top)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        default_conf = int(self.config.get("model", {}).get("conf", 0.25) * 100)
        self.conf_slider.setValue(default_conf)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        conf_layout.addWidget(self.conf_slider)
        self._update_conf_label(default_conf)

        # Çizim (Ekran)
        draw_conf_top = QHBoxLayout()
        draw_conf_top.addWidget(QLabel("Drawing (Display):"))
        draw_conf_top.addStretch()
        self._draw_conf_value_lbl = QLabel()
        self._draw_conf_value_lbl.setStyleSheet("color: #38BDF8; font-weight: 700;")
        draw_conf_top.addWidget(self._draw_conf_value_lbl)
        conf_layout.addLayout(draw_conf_top)

        self.draw_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.draw_conf_slider.setRange(0, 100)
        self.draw_conf_slider.setValue(default_conf)
        self.draw_conf_slider.valueChanged.connect(self._on_draw_conf_changed)
        conf_layout.addWidget(self.draw_conf_slider)
        self._update_draw_conf_label(default_conf)
        layout.addWidget(conf_grp)

        # ── Processing FPS Limit Group ─────────────────────────
        fps_limit_grp = QGroupBox("PROCESSING SPEED (FPS LIMIT)")
        fps_limit_layout = QVBoxLayout(fps_limit_grp)
        fps_limit_layout.setSpacing(6)
        
        fps_limit_top = QHBoxLayout()
        fps_limit_top.addWidget(QLabel("Maximum FPS:"))
        fps_limit_top.addStretch()
        self._fps_limit_lbl = QLabel("Unlimited")
        self._fps_limit_lbl.setStyleSheet("color: #38BDF8; font-weight: 700;")
        fps_limit_top.addWidget(self._fps_limit_lbl)
        fps_limit_layout.addLayout(fps_limit_top)
        
        self.fps_limit_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_limit_slider.setRange(0, 60) # 0 means unlimited
        self.fps_limit_slider.setValue(0)
        self.fps_limit_slider.valueChanged.connect(self._on_fps_limit_changed)
        fps_limit_layout.addWidget(self.fps_limit_slider)
        layout.addWidget(fps_limit_grp)

        # ── Path Manager Bar ───────────────────────────────────
        rec_dir = self.config.get("recording", {}).get("video_output_dir", "output/recordings")
        cap_dir = self.config.get("recording", {}).get("capture_output_dir", "data/raw_data/captured")
        self.path_manager = PathManagerBar(self.config_path, rec_dir, cap_dir)
        self.path_manager.config_saved.connect(self._on_path_config_saved)
        layout.addWidget(self.path_manager)

        layout.addStretch()

        # ── Action Buttons ─────────────────────────────────────
        self.btn_start = QPushButton("Start")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self.toggle_stream)
        layout.addWidget(self.btn_start)

        self.btn_record = QPushButton("Start Recording")
        self.btn_record.setObjectName("btnRecord")
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self.toggle_record)
        layout.addWidget(self.btn_record)

        self.btn_capture = QPushButton("Capture")
        self.btn_capture.setObjectName("btnCapture")
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self.request_capture)
        layout.addWidget(self.btn_capture)

        self.btn_help = QPushButton("Help & Guide")
        self.btn_help.setObjectName("btnHelp")
        self.btn_help.clicked.connect(self.show_help)
        layout.addWidget(self.btn_help)

        # Version
        ver_lbl = QLabel("v2.0 — Qt6 Edition")
        ver_lbl.setStyleSheet("color: #475569; font-size: 10px; margin-top: 8px;")
        ver_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ver_lbl)

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        return sidebar

    @staticmethod
    def _divider() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet("background-color: #1E2A3A; max-height: 1px; margin: 4px 0;")
        return f

    def _populate_models(self):
        models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models"
        all_models: list[Path] = []
        for ext in ("*.pt", "*.onnx", "*.engine"):
            all_models.extend(models_dir.rglob(ext))

        self._available_models = all_models
        if not all_models:
            self.model_combo.addItem("Model not found")
            return

        for p in all_models:
            badge, _, _ = _MODEL_TYPE_STYLES.get(p.suffix, ("?", "#fff", "#000"))
            rel = str(p.relative_to(models_dir))
            self.model_combo.addItem(f"[{badge}]  {rel}", userData=str(p))

        # Pre-select from config
        config_path = self.config.get("model", {}).get("path", "")
        if config_path:
            name = Path(config_path).name
            for i, p in enumerate(all_models):
                if p.name == name:
                    self.model_combo.setCurrentIndex(i)
                    break

    def _apply_initial_conf(self):
        self._on_conf_changed(self.conf_slider.value())
        self._on_draw_conf_changed(self.draw_conf_slider.value())

    # ═══════════════════════════════════════════════════════════
    #  Slots / Event Handlers
    # ═══════════════════════════════════════════════════════════

    def _on_conf_changed(self, value: int):
        conf = value / 100.0
        self._update_conf_label(value)
        if self._processor:
            self._processor.conf = conf

    def _update_conf_label(self, value: int):
        self._conf_value_lbl.setText(f"{value / 100:.2f}")

    def _on_draw_conf_changed(self, value: int):
        conf = value / 100.0
        self._update_draw_conf_label(value)
        if self._annotator:
            self._annotator.conf_threshold = conf

    def _update_draw_conf_label(self, value: int):
        self._draw_conf_value_lbl.setText(f"{value / 100:.2f}")

    def _on_fps_limit_changed(self, value: int):
        if value == 0:
            self._fps_limit_lbl.setText("Unlimited")
        else:
            self._fps_limit_lbl.setText(f"{value} FPS")
        if self._processor:
            self._processor.target_fps = value

    @pyqtSlot(bool)
    def _on_play_toggled(self, is_playing: bool):
        if self._source:
            if is_playing:
                self._source.resume()
            else:
                self._source.pause()

    @pyqtSlot(int)
    def _on_seek_requested(self, frame_idx: int):
        if self._source:
            self._source.seek(frame_idx)
            # Clear pipeline output queue to immediately show the new frame
            while not self.q_out.empty():
                try:
                    self.q_out.get_nowait()
                except queue.Empty:
                    break

    @pyqtSlot(float)
    def _on_playback_speed_changed(self, speed: float):
        if self._source and hasattr(self._source, "set_playback_speed"):
            self._source.set_playback_speed(speed)

    # --- Gallery & Image Navigation ---
    def _on_gallery_image_selected(self, index: int):
        if self._source and getattr(self._source, 'is_image_dir', lambda: False)():
            self._source.seek(index)
            
    def _on_prev_image(self):
        if self._source and getattr(self._source, 'is_image_dir', lambda: False)():
            pos = self._source.get_position()
            if pos > 0:
                self._source.seek(pos - 1)

    def _on_next_image(self):
        if self._source and getattr(self._source, 'is_image_dir', lambda: False)():
            pos = self._source.get_position()
            tot = self._source.get_total_frames()
            if pos < tot - 1:
                self._source.seek(pos + 1)

    def toggle_stream(self):
        if self.is_running:
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def toggle_record(self):
        if not self._recorder:
            return
        if self._recorder.is_recording():
            summary = self._recorder.stop()
            self.btn_record.setText("Start Recording")
            self.btn_record.setProperty("recording", "false")
            self.btn_record.style().polish(self.btn_record)
            logger.info("recording_stopped", summary=vars(summary))
        else:
            try:
                self._recorder.start()
                self.btn_record.setText("Stop Recording")
                self.btn_record.setProperty("recording", "true")
                self.btn_record.style().polish(self.btn_record)
            except Exception as e:
                logger.error("recorder_error", error=str(e))

    # ═══════════════════════════════════════════════════════════
    #  Pipeline Management
    # ═══════════════════════════════════════════════════════════

    def _start_pipeline(self):
        idx = self.model_combo.currentIndex()
        if idx < 0 or not self._available_models:
            self._status_bar.showMessage("No model selected!")
            return

        # Resolve selected model path
        model_path_str = self.model_combo.currentData()
        if model_path_str:
            model_path = Path(model_path_str)
        else:
            model_path = self._available_models[idx] if idx < len(self._available_models) else None

        if not model_path or not model_path.exists():
            self._status_bar.showMessage("Model file not found!")
            return

        # UI → loading state
        self.btn_start.setText("Loading...")
        self.btn_start.setEnabled(False)
        self._status_bar.showMessage(f"Loading model: {model_path.name}")

        # Load model in background thread
        self._loader_thread = _PipelineLoader(self.model_loader, model_path)
        self._loader_thread.loaded.connect(self._on_model_loaded)
        self._loader_thread.failed.connect(self._on_model_failed)
        self._loader_thread.start()

    @pyqtSlot(object, dict)
    def _on_model_loaded(self, model, class_names: dict):
        try:
            # Source
            src_type = self.source_selector.get_source_type()
            raw_src = self.source_selector.get_source()
            src = int(raw_src) if src_type == "camera" and raw_src.isdigit() else raw_src

            reconnect_cfg = self.config.get("source", {}).get("reconnect", {})
            max_retries = reconnect_cfg.get("max_retries", 10)
            retry_interval = reconnect_cfg.get("retry_interval_sec", 3.0)

            if src_type == "image_dir":
                self._source = ImageDirectorySource(
                    source=src,
                    max_queue_size=3
                )
                self._source.set_fps(self.fps_limit_slider.value() if self.fps_limit_slider.value() > 0 else 5)
            else:
                self._source = ResilientVideoSource(
                    source=src,
                    max_queue_size=3,
                    max_retries=max_retries,
                    retry_interval_sec=retry_interval
                )
            self._source.open()

            # Set up playback controls if file
            if getattr(self._source, 'is_image_dir', lambda: False)():
                self.playback_control.hide()
                self.image_gallery.show()
                self.image_gallery.set_images(self._source.get_image_list())
                self.video_widget.set_image_mode(True)
            elif self._source.is_file():
                self.playback_control.setup(self._source.get_total_frames(), self._source.get_fps())
                self.playback_control.show()
                self.image_gallery.hide()
                self.video_widget.set_image_mode(False)
            else:
                self.playback_control.hide()
                self.image_gallery.hide()
                self.video_widget.set_image_mode(False)

            # Annotator & Processor
            colors = generate_class_colors(class_names)
            conf = self.conf_slider.value() / 100.0
            draw_conf = self.draw_conf_slider.value() / 100.0
            self._annotator = Annotator(class_colors=colors, conf_threshold=draw_conf)
            
            # Determine thermal mode
            thermal_mode = self.thermal_control.get_colormap() if self.thermal_control.is_thermal() else None
            thermal_alpha = self.thermal_control.get_alpha()
            thermal_beta = self.thermal_control.get_beta()
            
            self._processor = FrameProcessor(
                model=model,
                input_queue=self._source.get_queue(),
                output_queue=self.q_out,
                conf=conf,
                device=_DEVICE,
                thermal_mode=thermal_mode,
                thermal_alpha=thermal_alpha,
                thermal_beta=thermal_beta,
            )
            self._processor.target_fps = self.fps_limit_slider.value()
            self._processor.start()

            # Recorder
            out_dir = self.path_manager.get_recording_dir()
            self._recorder = VideoRecorder(
                output_dir=out_dir,
                fps=self._source.get_fps(),
                resolution=self._source.get_resolution(),
            )

            self.is_running = True
            self.btn_start.setText("Stop")
            self.btn_start.setObjectName("btnStop")
            self.btn_start.style().polish(self.btn_start)
            self.btn_start.setEnabled(True)
            self.btn_record.setEnabled(True)
            self.btn_capture.setEnabled(True)
            self._status_bar.showMessage(f"Stream active  —  {_DEVICE_LABEL}")

            if self.config.get("recording", {}).get("enabled", False):
                self.toggle_record()

            logger.info("pipeline_started", device=_DEVICE)

        except SourceOpenError as e:
            logger.error("pipeline_source_error", error=str(e))
            self._on_model_failed(str(e))
        except Exception as e:
            logger.error("pipeline_start_failed", error=str(e))
            self._on_model_failed(str(e))

    @pyqtSlot(str)
    def _on_model_failed(self, error: str):
        self.btn_start.setText("Start")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.style().polish(self.btn_start)
        self.btn_start.setEnabled(True)
        self._status_bar.showMessage(f"Error: {error}")

    @pyqtSlot(int)
    def _on_model_combo_changed(self, index: int):
        model_path_str = self.model_combo.itemData(index)
        if model_path_str:
            model_name = Path(model_path_str).name
            self.thermal_control.auto_detect_model(model_name)

    @pyqtSlot(bool, str)
    def _on_thermal_mode_changed(self, is_thermal: bool, colormap: str):
        if self._processor:
            self._processor.thermal_mode = colormap if is_thermal else None
            if self._source and getattr(self._source, '_is_paused', False):
                self._processor.force_reprocess()
            logger.info("thermal_mode_changed_dynamically", is_thermal=is_thermal, colormap=colormap)

    @pyqtSlot(float, int)
    def _on_thermal_intensity_changed(self, alpha: float, beta: int):
        if self._processor:
            self._processor.thermal_alpha = alpha
            self._processor.thermal_beta = beta
            if self._source and getattr(self._source, '_is_paused', False):
                self._processor.force_reprocess()

    @pyqtSlot(str)
    def _on_path_config_saved(self, message: str):
        self._status_bar.showMessage(message, 4000)

    @pyqtSlot()
    def show_help(self):
        win = HelpWindow(self)
        win.exec()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right:
            if self._source and self._source.is_file():
                self._on_seek_requested(self._source.get_position() + 1)
        elif event.key() == Qt.Key.Key_Left:
            if self._source and self._source.is_file():
                self._on_seek_requested(max(0, self._source.get_position() - 1))
        super().keyPressEvent(event)


    def _stop_pipeline(self):
        self.is_running = False

        if self._recorder and self._recorder.is_recording():
            self._recorder.stop()

        if self._processor:
            self._processor.stop()
            self._processor = None

        if self._source:
            self._source.release()
            self._source = None

        # Drain queue
        while not self.q_out.empty():
            try:
                self.q_out.get_nowait()
            except Exception:
                break

        self.btn_start.setText("Start")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.style().polish(self.btn_start)
        self.btn_start.setEnabled(True)
        self.btn_record.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.btn_record.setText("Start Recording")

        self.playback_control.hide()
        self.image_gallery.hide()

        self.video_widget.clear_frame()
        self.stats_panel.reset()
        self.source_selector.update_status(ConnectionStatus.DISCONNECTED)
        self._status_bar.showMessage("Stopped.")
        logger.info("pipeline_stopped")

    # ═══════════════════════════════════════════════════════════
    #  Frame Update Loop (QTimer — 15ms)
    # ═══════════════════════════════════════════════════════════

    def request_capture(self):
        self._capture_requested = True

    def _update_frame(self):
        if not self.is_running or not self._annotator or not self._source:
            return

        try:
            result = None
            while not self.q_out.empty():
                result = self.q_out.get_nowait()

            # Her döngüde bağlantı durumunu güncelle
            conn_status = self._source._connection_status
            self.stats_panel.update_connection(conn_status.name)
            self.source_selector.update_status(conn_status)

            if getattr(self._source, 'is_image_dir', lambda: False)():
                pos = self._source.get_position()
                self.image_gallery.select_image(pos)
                info = self._source.get_current_image_details()
                tot = self._source.get_total_frames()
                self.video_widget.set_image_info(f"Image {pos + 1} / {tot}\n{info}")
            elif self._source.is_file():
                self.playback_control.update_position(self._source.get_position())
                if self._source._is_paused and self.playback_control.is_playing:
                    # Kaynak otomatik olarak durakladıysa (örneğin EOF'a ulaştıysa), UI'ı da duraklat
                    self.playback_control.is_playing = False
                    self.playback_control.btn_play_pause.setText("▶")
                    self.playback_control.play_toggled.emit(False)

            if result is not None and result.frame is not None and result.frame.size > 0:
                if getattr(self, '_capture_requested', False):
                    self._capture_requested = False
                    self._save_capture(result.frame)

                annotated = self._annotator.draw(result.frame, result)
                self._fps_counter.tick()
                fps = self._fps_counter.get_fps()

                if self._recorder and self._recorder.is_recording():
                    self._recorder.write(annotated)

                self.video_widget.render_frame(annotated)
                self.stats_panel.update_fps(fps)
                self.stats_panel.update_inference(result.inference_ms)
                self.stats_panel.update_detections(len(result.detections))
                self.stats_panel.update_src_fps(self._source.get_fps())

            else:
                # Show raw frame if no processed result yet
                try:
                    raw_ret, raw_frame = self._source.get_queue().get_nowait()
                    if raw_ret and raw_frame is not None and raw_frame.size > 0:
                        if getattr(self, '_capture_requested', False):
                            self._capture_requested = False
                            self._save_capture(raw_frame)

                        self._fps_counter.tick()
                        fps = self._fps_counter.get_fps()

                        self.video_widget.render_frame(raw_frame)
                        self.stats_panel.update_fps(fps)
                        self.stats_panel.update_src_fps(self._source.get_fps())
                except queue.Empty:
                    pass

        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════
    #  Window Lifecycle
    # ═══════════════════════════════════════════════════════════

    def _save_capture(self, frame):
        if frame is None or frame.size == 0:
            return
        save_dir = self.path_manager.get_capture_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        cv2.imwrite(str(save_dir / filename), frame)
        self._status_bar.showMessage(f"Veri yakalandı: {filename}", 3000)

    def closeEvent(self, event):
        self._timer.stop()
        if self.is_running:
            self._stop_pipeline()
        event.accept()
