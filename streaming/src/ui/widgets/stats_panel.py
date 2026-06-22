"""StatsPanel — Real-time inference statistics display widget."""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class StatBlockWidget(QWidget):
    """A custom widget representing a single stat display block with dynamic font scaling."""
    
    def __init__(self, label_text: str, unit: str, parent=None):
        super().__init__(parent)
        self.setObjectName("statBlock")
        self.setStyleSheet("""
            QWidget#statBlock {
                background-color: #1A2233;
                border: 1px solid #1E2A3A;
                border-radius: 8px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.value_lbl = QLabel("—")
        self.value_lbl.setObjectName("statValue")
        self.value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_lbl.setStyleSheet("color: #38BDF8;")

        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setObjectName("statUnit")
        self.unit_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.unit_lbl.setStyleSheet("color: #475569; letter-spacing: 0.5px;")

        self.name_lbl = QLabel(label_text)
        self.name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_lbl.setStyleSheet("color: #64748B; font-weight: 600; letter-spacing: 0.5px;")

        layout.addWidget(self.value_lbl)
        layout.addWidget(self.unit_lbl)
        layout.addWidget(self.name_lbl)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Dynamically scale font size based on the height of the widget
        h = self.height()
        val_size = max(11, int(h * 0.22))
        unit_size = max(7, int(h * 0.09))
        name_size = max(8, int(h * 0.1))
        
        f_val = self.value_lbl.font()
        f_val.setPointSize(val_size)
        f_val.setBold(True)
        self.value_lbl.setFont(f_val)
        
        f_unit = self.unit_lbl.font()
        f_unit.setPointSize(unit_size)
        self.unit_lbl.setFont(f_unit)
        
        f_name = self.name_lbl.font()
        f_name.setPointSize(name_size)
        f_name.setBold(True)
        self.name_lbl.setFont(f_name)


class StatsPanel(QWidget):
    """Horizontal stats strip: FPS | Inference ms | Detections | Source FPS | Connection | GPU."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statsPanel")
        self.setMinimumHeight(85)
        self.setStyleSheet("background-color: #0F1117; border-top: 1px solid #1E2A3A;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self.fps_widget = StatBlockWidget("PROCESS SPEED", "FPS")
        self.inf_widget = StatBlockWidget("LATENCY", "ms")
        self.det_widget = StatBlockWidget("DETECTION", "count")
        self.src_fps_widget = StatBlockWidget("SOURCE SPEED", "FPS")
        self.conn_widget = StatBlockWidget("CONNECTION", "status")
        
        self._fps_lbl = self.fps_widget.value_lbl
        self._inf_lbl = self.inf_widget.value_lbl
        self._det_lbl = self.det_widget.value_lbl
        self._src_fps_lbl = self.src_fps_widget.value_lbl
        self._conn_lbl = self.conn_widget.value_lbl
        
        # Color labels differently
        self._inf_lbl.setStyleSheet("color: #A78BFA;")
        self._det_lbl.setStyleSheet("color: #34D399;")
        self._src_fps_lbl.setStyleSheet("color: #F43F5E;")  # Rose color
        self._conn_lbl.setStyleSheet("color: #475569;")   # Slate/grey initially

        # Adjust font sizes for status value
        self._conn_lbl.setFont(QFont("Inter", 13, QFont.Weight.Bold))

        layout.addWidget(self.fps_widget, stretch=1)
        layout.addWidget(self._sep())
        layout.addWidget(self.inf_widget, stretch=1)
        layout.addWidget(self._sep())
        layout.addWidget(self.det_widget, stretch=1)
        layout.addWidget(self._sep())
        layout.addWidget(self.src_fps_widget, stretch=1)
        layout.addWidget(self._sep())
        layout.addWidget(self.conn_widget, stretch=1)

        # GPU label (compact right-side)
        self._gpu_lbl = QLabel("GPU: —")
        self._gpu_lbl.setStyleSheet("""
            color: #64748B;
            font-size: 9px;
            padding: 2px 6px;
            border-left: 1px solid #1E2A3A;
        """)
        self._gpu_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._gpu_lbl)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Dynamically scale GPU label font size based on height
        h = self.height()
        gpu_size = max(8, int(h * 0.1))
        f_gpu = self._gpu_lbl.font()
        f_gpu.setPointSize(gpu_size)
        self._gpu_lbl.setFont(f_gpu)

    @staticmethod
    def _sep() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("background-color: #1E2A3A; max-width: 1px;")
        return sep

    # ── Public update methods ─────────────────────────────────

    def update_fps(self, fps: float) -> None:
        color = "#38BDF8" if fps >= 20 else ("#FBBF24" if fps >= 10 else "#EF4444")
        self._fps_lbl.setStyleSheet(f"color: {color};")
        self._fps_lbl.setText(f"{fps:.1f}")

    def update_inference(self, ms: float) -> None:
        color = "#A78BFA" if ms < 50 else ("#FBBF24" if ms < 150 else "#EF4444")
        self._inf_lbl.setStyleSheet(f"color: {color};")
        self._inf_lbl.setText(f"{ms:.1f}")

    def update_detections(self, count: int) -> None:
        self._det_lbl.setText(str(count))

    def update_src_fps(self, fps: float) -> None:
        self._src_fps_lbl.setText(f"{fps:.1f}")

    def update_connection(self, status_str: str) -> None:
        """Bağlantı durumuna göre istatistik panelini günceller."""
        s = status_str.lower()
        if "connected" in s:
            self._conn_lbl.setText("CONNECTED")
            self._conn_lbl.setStyleSheet("color: #22C55E;")  # Yeşil
        elif "reconnecting" in s:
            self._conn_lbl.setText("RETRY")
            self._conn_lbl.setStyleSheet("color: #EAB308;")  # Sarı
        elif "failed" in s:
            self._conn_lbl.setText("ERROR")
            self._conn_lbl.setStyleSheet("color: #EF4444;")  # Kırmızı
        else:
            self._conn_lbl.setText("NONE")
            self._conn_lbl.setStyleSheet("color: #475569;")  # Gri

    def update_gpu_label(self, text: str) -> None:
        self._gpu_lbl.setText(text)

    def reset(self) -> None:
        self._fps_lbl.setText("—")
        self._inf_lbl.setText("—")
        self._det_lbl.setText("—")
        self._src_fps_lbl.setText("—")
        self._conn_lbl.setText("—")
        self._fps_lbl.setStyleSheet("color: #38BDF8;")
        self._inf_lbl.setStyleSheet("color: #A78BFA;")
        self._det_lbl.setStyleSheet("color: #34D399;")
        self._conn_lbl.setStyleSheet("color: #475569;")

