import yaml
from pathlib import Path
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog
)
from PyQt6.QtCore import pyqtSlot, pyqtSignal

class PathManagerBar(QGroupBox):
    """Video kayıt ve yakalanan görsel dizinlerini yöneten panel."""
    paths_changed = pyqtSignal(str, str)  # (rec_dir, cap_dir)
    config_saved = pyqtSignal(str)  # Durum mesajı fırlatır

    def __init__(self, config_path: Path | str, default_rec_dir: str, default_cap_dir: str, parent=None):
        super().__init__("DOSYA YOLLARI YÖNETİMİ", parent)
        self.config_path = Path(config_path)
        self._rec_dir = default_rec_dir
        self._cap_dir = default_cap_dir
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 14, 10, 10)

        # ── Video Kayıt Dizini ──
        rec_lbl = QLabel("Video Recording Folder:")
        rec_lbl.setStyleSheet("color: #64748B; font-size: 10px; font-weight: bold;")
        layout.addWidget(rec_lbl)

        rec_row = QHBoxLayout()
        self.rec_edit = QLineEdit(self._rec_dir)
        self.rec_edit.setReadOnly(True)
        rec_row.addWidget(self.rec_edit)

        rec_browse = QPushButton("Select")
        rec_browse.setStyleSheet("""
            QPushButton {
                padding: 4px 6px;
                min-height: 28px;
                font-size: 11px;
            }
        """)
        rec_browse.setFixedWidth(55)
        rec_browse.clicked.connect(self._browse_rec)
        rec_row.addWidget(rec_browse)
        layout.addLayout(rec_row)

        # ── Capture Görsel Dizini ──
        cap_lbl = QLabel("Capture Folder:")
        cap_lbl.setStyleSheet("color: #64748B; font-size: 10px; font-weight: bold;")
        layout.addWidget(cap_lbl)

        cap_row = QHBoxLayout()
        self.cap_edit = QLineEdit(self._cap_dir)
        self.cap_edit.setReadOnly(True)
        cap_row.addWidget(self.cap_edit)

        cap_browse = QPushButton("Select")
        cap_browse.setStyleSheet("""
            QPushButton {
                padding: 4px 6px;
                min-height: 28px;
                font-size: 11px;
            }
        """)
        cap_browse.setFixedWidth(55)
        cap_browse.clicked.connect(self._browse_cap)
        cap_row.addWidget(cap_browse)
        layout.addLayout(cap_row)

        # ── Ayarları Kaydet Butonu ──
        self.save_btn = QPushButton("Save as Default")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E293B;
                border: 1px solid #334155;
                font-size: 11px;
                color: #CBD5E1;
                margin-top: 6px;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #1E3A8A;
                border-color: #3B82F6;
                color: #FFFFFF;
            }
        """)
        self.save_btn.clicked.connect(self.save_to_yaml)
        layout.addWidget(self.save_btn)

    def get_recording_dir(self) -> Path:
        return Path(self.rec_edit.text())

    def get_capture_dir(self) -> Path:
        return Path(self.cap_edit.text())

    @pyqtSlot()
    def _browse_rec(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Video Recording Folder", self.rec_edit.text())
        if dir_path:
            self.rec_edit.setText(dir_path)
            self.paths_changed.emit(dir_path, self.cap_edit.text())

    @pyqtSlot()
    def _browse_cap(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Capture Folder", self.cap_edit.text())
        if dir_path:
            self.cap_edit.setText(dir_path)
            self.paths_changed.emit(self.rec_edit.text(), dir_path)

    def save_to_yaml(self):
        """Ayarları YAML konfigürasyon dosyasına kalıcı olarak yazar."""
        try:
            if not self.config_path.exists():
                self.config_saved.emit("Error: YAML file not found!")
                return

            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Dizin yollarını güncelle
            if "recording" not in data:
                data["recording"] = {}
            data["recording"]["video_output_dir"] = self.rec_edit.text()
            data["recording"]["capture_output_dir"] = self.cap_edit.text()

            # Geriye dönük uyumluluk için eski anahtarı da senkronize edelim
            data["recording"]["output_dir"] = self.rec_edit.text()

            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

            self.config_saved.emit("Settings saved successfully!")
        except Exception as e:
            self.config_saved.emit(f"Hata: {str(e)}")
