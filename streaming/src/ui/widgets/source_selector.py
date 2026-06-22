from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QLineEdit, QPushButton, QFileDialog, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QBrush
from pathlib import Path
from ...core.exceptions import ConnectionStatus

class LEDIndicator(QWidget):
    """Küçük renkli LED durum göstergesi."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self._color = QColor("#475569")  # Varsayılan gri

    def set_color(self, color_hex: str):
        self._color = QColor(color_hex)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._color))
        painter.drawEllipse(0, 0, self.width(), self.height())


class SourceSelectorWidget(QGroupBox):
    """Çoklu kaynak seçimi ve bağlantı durum göstergesini yöneten QGroupBox."""
    source_changed = pyqtSignal(str, str)  # (type, value)

    def __init__(self, parent=None):
        super().__init__("SOURCE SELECTOR", parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 14, 10, 10)

        # ── Kaynak Türü ──
        type_lbl = QLabel("Source Type:")
        type_lbl.setStyleSheet("color: #64748B; font-size: 10px; font-weight: bold;")
        layout.addWidget(type_lbl)

        self.type_combo = QComboBox()
        self.type_combo.addItem("Webcam (Local)", "camera")
        self.type_combo.addItem("Image Folder (.jpg, .png)", "image_dir")
        self.type_combo.addItem("Local Video (.mp4)", "video")
        self.type_combo.addItem("IP Camera (HTTP)", "ip")
        self.type_combo.addItem("Drone Stream (RTSP)", "rtsp")
        self.type_combo.addItem("RTMP Stream", "rtmp")
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        layout.addWidget(self.type_combo)

        # ── Kaynak Değeri / Adresi ──
        addr_lbl = QLabel("Source Address / Path:")
        addr_lbl.setStyleSheet("color: #64748B; font-size: 10px; font-weight: bold;")
        layout.addWidget(addr_lbl)

        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("0")
        self.value_edit.textChanged.connect(self._on_value_changed)
        input_row.addWidget(self.value_edit)

        self.browse_btn = QPushButton("...")
        self.browse_btn.setFixedWidth(36)
        self.browse_btn.setToolTip("Select local video file")
        self.browse_btn.clicked.connect(self._browse_file)
        self.browse_btn.setVisible(False)
        input_row.addWidget(self.browse_btn)

        layout.addLayout(input_row)

        # ── Bağlantı Durumu ──
        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        status_row.setContentsMargins(2, 4, 2, 2)

        self.led = LEDIndicator()
        status_row.addWidget(self.led)

        self.status_lbl = QLabel("Disconnected")
        self.status_lbl.setStyleSheet("color: #64748B; font-size: 11px; font-weight: 600;")
        status_row.addWidget(self.status_lbl)
        status_row.addStretch()

        layout.addLayout(status_row)

    def get_source(self) -> str:
        """Kullanıcının girdiği temizlenmiş kaynak metnini döner."""
        return self.value_edit.text().strip()

    def get_source_type(self) -> str:
        """Seçili kaynak türünün kod adını döner (camera, video, ip, rtsp, rtmp)."""
        return self.type_combo.currentData()

    @pyqtSlot(int)
    def _on_type_changed(self, index: int):
        type_str = self.type_combo.currentData()
        
        # UI güncellemeleri
        if type_str == "camera":
            self.value_edit.setPlaceholderText("0  (Default)")
            self.value_edit.setText("0")
            self.browse_btn.setVisible(False)
        elif type_str == "image_dir":
            self.value_edit.setPlaceholderText("Select image folder (e.g. data/raw_data/captured)...")
            self.value_edit.setText("")
            self.browse_btn.setVisible(True)
            self.browse_btn.setToolTip("Select image folder")
        elif type_str == "video":
            self.value_edit.setPlaceholderText("Select file path...")
            self.value_edit.setText("")
            self.browse_btn.setVisible(True)
            self.browse_btn.setToolTip("Select local video file")
        elif type_str == "ip":
            self.value_edit.setPlaceholderText("http://192.168.1.50:8080/video")
            self.value_edit.setText("")
            self.browse_btn.setVisible(False)
        elif type_str == "rtsp":
            self.value_edit.setPlaceholderText("rtsp://192.168.1.100:8554/live")
            self.value_edit.setText("")
            self.browse_btn.setVisible(False)
        elif type_str == "rtmp":
            self.value_edit.setPlaceholderText("rtmp://live.twitch.tv/app/...")
            self.value_edit.setText("")
            self.browse_btn.setVisible(False)

        self.source_changed.emit(type_str, self.value_edit.text())

    @pyqtSlot(str)
    def _on_value_changed(self, text: str):
        self.source_changed.emit(self.get_source_type(), text)

    @pyqtSlot()
    def _browse_file(self):
        type_str = self.type_combo.currentData()
        if type_str == "image_dir":
            dir_path = QFileDialog.getExistingDirectory(
                self, "Select Image Folder", ""
            )
            if dir_path:
                self.value_edit.setText(dir_path)
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
            )
            if file_path:
                self.value_edit.setText(file_path)

    def set_source(self, type_str: str, value: str):
        """Konfigürasyondan gelen varsayılan değerleri yükler."""
        idx = self.type_combo.findData(type_str)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)
        self.value_edit.setText(value)

    def update_status(self, status: ConnectionStatus, retry_count: int = 0, max_retries: int = 10):
        """Bağlantı durumuna göre LED ve durum etiketini günceller."""
        if status == ConnectionStatus.CONNECTED:
            self.led.set_color("#22C55E")  # Yeşil
            self.status_lbl.setText("CONNECTED")
            self.status_lbl.setStyleSheet("color: #22C55E; font-size: 11px; font-weight: bold;")
        elif status == ConnectionStatus.RECONNECTING:
            self.led.set_color("#EAB308")  # Sarı
            text = "RECONNECTING..."
            if retry_count > 0:
                text += f" ({retry_count}/{max_retries})"
            self.status_lbl.setText(text)
            self.status_lbl.setStyleSheet("color: #EAB308; font-size: 11px; font-weight: bold;")
        elif status == ConnectionStatus.FAILED:
            self.led.set_color("#EF4444")  # Kırmızı
            self.status_lbl.setText("CONNECTION ERROR")
            self.status_lbl.setStyleSheet("color: #EF4444; font-size: 11px; font-weight: bold;")
        elif status == ConnectionStatus.DISCONNECTED:
            self.led.set_color("#475569")  # Gri
            self.status_lbl.setText("Disconnected")
            self.status_lbl.setStyleSheet("color: #64748B; font-size: 11px; font-weight: 600;")
