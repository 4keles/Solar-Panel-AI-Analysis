from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import pyqtSignal, pyqtSlot

class ThermalControlWidget(QGroupBox):
    """Görüntüleme modunu (RGB / Termal) ve termal renk haritalarını yöneten panel."""
    mode_changed = pyqtSignal(bool, str)  # (is_thermal, colormap_name)
    intensity_changed = pyqtSignal(float, int) # (alpha, beta)

    def __init__(self, parent=None):
        super().__init__("GÖRÜNTÜ MODU", parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 14, 10, 10)

        # ── Mode Selection (Radio Buttons) ──
        self.btn_group = QButtonGroup(self)
        self.rgb_radio = QRadioButton("RGB Standard")
        self.thermal_radio = QRadioButton("Thermal Analysis")
        
        self.btn_group.addButton(self.rgb_radio)
        self.btn_group.addButton(self.thermal_radio)
        
        self.rgb_radio.setChecked(True)  # Default RGB

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.rgb_radio)
        radio_layout.addWidget(self.thermal_radio)
        layout.addLayout(radio_layout)

        # ── Thermal Colormap ──
        cmap_lbl = QLabel("Thermal Colormap:")
        cmap_lbl.setStyleSheet("color: #64748B; font-size: 10px; font-weight: bold;")
        layout.addWidget(cmap_lbl)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItem("Grayscale (Standard)", "grayscale")
        self.cmap_combo.addItem("Inferno (Industrial)", "inferno")
        self.cmap_combo.addItem("Jet (Traditional)", "jet")
        self.cmap_combo.addItem("Magma", "magma")
        self.cmap_combo.addItem("Hot (High Contrast)", "hot")
        self.cmap_combo.setEnabled(False)  # Disabled by default
        layout.addWidget(self.cmap_combo)

        # ── Contrast & Brightness ──
        self.contrast_lbl = QLabel("Contrast (Intensity): 1.0")
        self.contrast_lbl.setStyleSheet("color: #64748B; font-size: 10px;")
        layout.addWidget(self.contrast_lbl)

        from PyQt6.QtWidgets import QSlider
        from PyQt6.QtCore import Qt

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(5, 30) # 0.5x to 3.0x
        self.contrast_slider.setValue(10) # 1.0x
        layout.addWidget(self.contrast_slider)

        self.brightness_lbl = QLabel("Brightness: 0")
        self.brightness_lbl.setStyleSheet("color: #64748B; font-size: 10px;")
        layout.addWidget(self.brightness_lbl)

        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        layout.addWidget(self.brightness_slider)

        # Sinyal bağlantıları
        self.rgb_radio.toggled.connect(self._on_mode_toggled)
        self.thermal_radio.toggled.connect(self._on_mode_toggled)
        self.cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        self.contrast_slider.valueChanged.connect(self._on_intensity_changed)
        self.brightness_slider.valueChanged.connect(self._on_intensity_changed)

    def is_thermal(self) -> bool:
        """Termal modun seçili olup olmadığını döner."""
        return self.thermal_radio.isChecked()

    def get_colormap(self) -> str:
        """Seçili renk haritasını döner (grayscale, inferno, etc.)."""
        return self.cmap_combo.currentData()

    def get_alpha(self) -> float:
        return self.contrast_slider.value() / 10.0

    def get_beta(self) -> int:
        return self.brightness_slider.value()

    @pyqtSlot(bool)
    def _on_mode_toggled(self, checked: bool):
        # Yalnızca check olan radio button tetiklendiğinde işlem yap
        if checked:
            is_thermal = self.thermal_radio.isChecked()
            self.cmap_combo.setEnabled(is_thermal)
            self.mode_changed.emit(is_thermal, self.get_colormap())

    @pyqtSlot(int)
    def _on_cmap_changed(self, index: int):
        if self.is_thermal():
            self.mode_changed.emit(True, self.get_colormap())

    @pyqtSlot(int)
    def _on_intensity_changed(self, value: int):
        self.contrast_lbl.setText(f"Contrast (Intensity): {self.get_alpha():.1f}")
        self.brightness_lbl.setText(f"Brightness: {self.get_beta()}")
        self.intensity_changed.emit(self.get_alpha(), self.get_beta())

    def auto_detect_model(self, model_name: str):
        """Model dosya adına göre termal modu otomatik tespit edip değiştirir."""
        m_name_lower = model_name.lower()
        if "thermal" in m_name_lower or "gray" in m_name_lower or "termal" in m_name_lower:
            # Termal model tespit edildi, modu termale çek ve varsayılan grayscale yap
            self.thermal_radio.setChecked(True)
            idx = self.cmap_combo.findData("grayscale")
            if idx >= 0:
                self.cmap_combo.setCurrentIndex(idx)
        else:
            # Standart RGB model
            self.rgb_radio.setChecked(True)
