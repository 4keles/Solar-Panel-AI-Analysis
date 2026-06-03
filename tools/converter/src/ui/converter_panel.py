"""Converter Panel — Qt6 GUI for .pt → Engine/ONNX/TorchScript conversion."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QCheckBox,
    QLineEdit, QTextEdit, QFileDialog, QGroupBox,
    QProgressBar, QButtonGroup, QRadioButton, QSizePolicy,
    QSpinBox, QStatusBar, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QTextCursor

# Add src to path so converter_core is importable
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from converter_core import ModelConverter, ConversionConfig, ConversionResult, OutputFormat

# ─────────────────────────────────────────────────────────────
#  Background Worker
# ─────────────────────────────────────────────────────────────

class _ConversionWorker(QThread):
    log_msg    = pyqtSignal(str)
    finished   = pyqtSignal(object)  # ConversionResult

    def __init__(self, cfg: ConversionConfig):
        super().__init__()
        self._cfg = cfg

    def run(self):
        converter = ModelConverter(log_callback=self.log_msg.emit)
        result = converter.convert(self._cfg)
        self.finished.emit(result)


# ─────────────────────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────────────────────

class ConverterPanel(QMainWindow):
    """Standalone model converter GUI."""

    def __init__(self, models_dir: Path | None = None):
        super().__init__()
        self._models_dir = models_dir or Path("models")
        self._worker: _ConversionWorker | None = None
        self._selected_input: Path | None = None

        self.setWindowTitle("⚙  Solar Panel OD — Model Dönüştürücü")
        self.setMinimumSize(900, 680)
        self.resize(1000, 750)

        self._build_ui()
        self._populate_models()
        self._detect_system()

    # ═══════════════════════════════════════════════════════════
    #  UI Construction
    # ═══════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left config panel
        left = self._build_left_panel()
        main_layout.addWidget(left)

        # Right log panel
        right = self._build_right_panel()
        main_layout.addWidget(right, stretch=1)

        # Status bar
        self._status_bar = QStatusBar()
        self._status_bar.setFixedHeight(26)
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Model seçin ve dönüştürme parametrelerini ayarlayın.")

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("sidebar")
        panel.setFixedWidth(340)
        panel.setStyleSheet("QWidget#sidebar { background-color: #161B27; border-right: 1px solid #1E2A3A; }")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(12)

        # Title
        title = QLabel("⚙  Model Dönüştürücü")
        font = QFont(); font.setPointSize(14); font.setBold(True)
        title.setFont(font)
        title.setStyleSheet("color: #F1F5F9;")
        layout.addWidget(title)

        sub = QLabel(".pt  →  TensorRT Engine  /  ONNX  /  TorchScript")
        sub.setStyleSheet("color: #475569; font-size: 11px;")
        layout.addWidget(sub)
        layout.addWidget(self._sep())

        # ── System Info ──
        self._sys_info_lbl = QLabel("Sistem bilgisi alınıyor...")
        self._sys_info_lbl.setWordWrap(True)
        self._sys_info_lbl.setStyleSheet("""
            background-color: #1A2233; border: 1px solid #1E2A3A;
            border-radius: 6px; padding: 8px; font-size: 10px; color: #64748B;
        """)
        layout.addWidget(self._sys_info_lbl)

        # ── Input Model ──
        in_grp = QGroupBox("GİRDİ MODELİ (.pt)")
        in_grp.setStyleSheet(self._grp_style())
        in_layout = QVBoxLayout(in_grp)
        in_layout.setSpacing(6)

        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        in_layout.addWidget(self.model_combo)

        browse_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Veya dosya yolu girin...")
        self._path_edit.setReadOnly(True)
        self._path_edit.setStyleSheet("font-size: 10px; color: #64748B;")
        browse_row.addWidget(self._path_edit, stretch=1)

        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(34)
        browse_btn.clicked.connect(self._browse_model)
        browse_row.addWidget(browse_btn)
        in_layout.addLayout(browse_row)
        layout.addWidget(in_grp)

        # ── Output Format ──
        fmt_grp = QGroupBox("HEDEF FORMAT")
        fmt_grp.setStyleSheet(self._grp_style())
        fmt_layout = QVBoxLayout(fmt_grp)
        fmt_layout.setSpacing(8)

        self._fmt_group = QButtonGroup(self)
        formats = [
            ("⚡  TensorRT Engine (.engine)", OutputFormat.TENSORRT,
             "En yüksek hız — GPU gerektirir, TensorRT kurulu olmalı"),
            ("🔷  ONNX (.onnx)", OutputFormat.ONNX,
             "Evrensel format — CPU ve GPU'da çalışır"),
            ("📜  TorchScript (.torchscript)", OutputFormat.TORCHSCRIPT,
             "PyTorch tabanlı — en geniş uyumluluk"),
        ]

        for i, (label, fmt, hint) in enumerate(formats):
            rb = QRadioButton(label)
            rb.setProperty("format", fmt)
            rb.setStyleSheet("color: #CBD5E1; font-size: 12px;")
            if i == 0:
                rb.setChecked(True)
            self._fmt_group.addButton(rb, i)
            fmt_layout.addWidget(rb)

            hint_lbl = QLabel(f"  {hint}")
            hint_lbl.setStyleSheet("color: #475569; font-size: 10px; padding-left: 20px;")
            fmt_layout.addWidget(hint_lbl)

        layout.addWidget(fmt_grp)

        # ── Parameters ──
        par_grp = QGroupBox("PARAMETRELER")
        par_grp.setStyleSheet(self._grp_style())
        par_layout = QVBoxLayout(par_grp)
        par_layout.setSpacing(8)

        # imgsz
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Görüntü Boyutu:"))
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "416", "640", "800", "1024", "1280"])
        self.imgsz_combo.setCurrentText("640")
        row1.addWidget(self.imgsz_combo)
        par_layout.addLayout(row1)

        # half precision
        self.half_cb = QCheckBox("FP16 (Half Precision) — GPU'da 2x hız")
        self.half_cb.setChecked(True)
        self.half_cb.setStyleSheet("color: #CBD5E1;")
        par_layout.addWidget(self.half_cb)

        # simplify (ONNX only)
        self.simplify_cb = QCheckBox("ONNX Simplify (onnxsim)")
        self.simplify_cb.setChecked(True)
        self.simplify_cb.setStyleSheet("color: #CBD5E1;")
        par_layout.addWidget(self.simplify_cb)

        # batch
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Batch:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(1)
        self.batch_spin.setStyleSheet("background:#1E293B; border:1px solid #334155; border-radius:4px; padding:3px;")
        row3.addWidget(self.batch_spin)
        row3.addStretch()
        par_layout.addLayout(row3)

        # workspace
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("TRT Workspace (GB):"))
        self.workspace_spin = QSpinBox()
        self.workspace_spin.setRange(1, 16)
        self.workspace_spin.setValue(4)
        self.workspace_spin.setStyleSheet("background:#1E293B; border:1px solid #334155; border-radius:4px; padding:3px;")
        row4.addWidget(self.workspace_spin)
        row4.addStretch()
        par_layout.addLayout(row4)

        layout.addWidget(par_grp)

        # ── Output Dir ──
        out_grp = QGroupBox("ÇIKTI DİZİNİ")
        out_grp.setStyleSheet(self._grp_style())
        out_layout = QHBoxLayout(out_grp)
        self._out_edit = QLineEdit()
        self._out_edit.setText(str(self._models_dir))
        out_layout.addWidget(self._out_edit, stretch=1)
        out_browse = QPushButton("…")
        out_browse.setFixedWidth(34)
        out_browse.clicked.connect(self._browse_output)
        out_layout.addWidget(out_browse)
        layout.addWidget(out_grp)

        layout.addStretch()

        # ── Convert Button ──
        self.btn_convert = QPushButton("⚡  Dönüştür")
        self.btn_convert.setObjectName("btnStart")
        self.btn_convert.setMinimumHeight(44)
        self.btn_convert.clicked.connect(self.start_conversion)
        layout.addWidget(self.btn_convert)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background-color: #0F1117;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Header
        hdr = QLabel("📋  Dönüştürme Logu")
        font = QFont(); font.setPointSize(12); font.setBold(True)
        hdr.setFont(font)
        hdr.setStyleSheet("color: #94A3B8;")
        layout.addWidget(hdr)

        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #0B0F18;
                color: #94A3B8;
                border: 1px solid #1E2A3A;
                border-radius: 8px;
                font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
                font-size: 12px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.log_area, stretch=1)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setStyleSheet("""
            QProgressBar { border: none; background-color: #1E293B; border-radius: 3px; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #1D4ED8, stop:1 #38BDF8); border-radius: 3px; }
        """)
        layout.addWidget(self._progress)

        # Result badge area
        self._result_lbl = QLabel("")
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setMinimumHeight(40)
        self._result_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._result_lbl)

        return panel

    # ═══════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _grp_style() -> str:
        return """
            QGroupBox {
                background-color: #1A2233; border: 1px solid #1E2A3A;
                border-radius: 8px; margin-top: 8px; padding: 10px 8px 8px 8px;
                font-size: 10px; font-weight: 600; color: #64748B;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 0 6px; left: 10px; color: #64748B;
                letter-spacing: 1px; font-size: 10px;
            }
        """

    @staticmethod
    def _sep() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet("background-color: #1E2A3A; max-height: 1px; margin: 2px 0;")
        return f

    def _populate_models(self):
        self.model_combo.clear()
        self.model_combo.addItem("— Model seçin —", userData=None)
        if self._models_dir.exists():
            for p in sorted(self._models_dir.rglob("*.pt")):
                rel = str(p.relative_to(self._models_dir))
                self.model_combo.addItem(f"[PT]  {rel}", userData=str(p))

    def _detect_system(self):
        try:
            converter = ModelConverter()
            info = converter.detect_system()

            lines = []
            torch_ver = info.get("torch", "—")
            lines.append(f"PyTorch: {torch_ver}")

            if info.get("cuda_available"):
                lines.append(f"GPU: {info.get('gpu_name', '?')}")
                lines.append(f"CUDA: {info.get('cuda_version', '?')}  |  VRAM: {info.get('vram_gb', '?')} GB")
            else:
                lines.append("⚠ CUDA yok — CPU modu")

            trt = info.get("tensorrt")
            lines.append(f"TensorRT: {'✓ ' + trt if trt else '✗ kurulu değil'}")

            onnx = info.get("onnx")
            ort = info.get("onnxruntime")
            lines.append(f"ONNX: {'✓ ' + onnx if onnx else '✗'} | ORT: {'✓ ' + ort if ort else '✗'}")

            self._sys_info_lbl.setText("\n".join(lines))
        except Exception as e:
            self._sys_info_lbl.setText(f"Sistem bilgisi alınamadı: {e}")

    # ═══════════════════════════════════════════════════════════
    #  Actions
    # ═══════════════════════════════════════════════════════════

    def _on_model_selected(self, idx: int):
        path_str = self.model_combo.currentData()
        if path_str:
            self._selected_input = Path(path_str)
            self._path_edit.setText(path_str)
        else:
            self._selected_input = None
            self._path_edit.clear()

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Model Seç", str(self._models_dir),
            "Model Dosyaları (*.pt *.onnx);;Tüm Dosyalar (*)"
        )
        if path:
            self._selected_input = Path(path)
            self._path_edit.setText(path)
            # Add to combo if not already there
            if self.model_combo.findData(path) == -1:
                self.model_combo.addItem(f"[PT]  {Path(path).name}", userData=path)
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Çıktı Dizini Seç", self._out_edit.text())
        if path:
            self._out_edit.setText(path)

    def start_conversion(self):
        if self._worker and self._worker.isRunning():
            self._append_log("[WARNING] Dönüştürme zaten devam ediyor...")
            return

        if not self._selected_input or not self._selected_input.exists():
            self._result_lbl.setText("⚠  Lütfen geçerli bir .pt model seçin.")
            self._result_lbl.setStyleSheet("color: #EF4444;")
            return

        # Resolve selected format
        checked_btn = self._fmt_group.checkedButton()
        fmt: OutputFormat = checked_btn.property("format")

        cfg = ConversionConfig(
            input_path=self._selected_input,
            output_dir=Path(self._out_edit.text()),
            format=fmt,
            imgsz=int(self.imgsz_combo.currentText()),
            half=self.half_cb.isChecked(),
            batch=self.batch_spin.value(),
            workspace_gb=self.workspace_spin.value(),
            simplify=self.simplify_cb.isChecked(),
        )

        # Clear previous log
        self.log_area.clear()
        self._result_lbl.clear()
        self._progress.setVisible(True)
        self.btn_convert.setEnabled(False)
        self.btn_convert.setText("⌛  Dönüştürülüyor...")
        self._status_bar.showMessage(f"Dönüştürme: {cfg.input_path.name} → {cfg.format.value}")

        self._worker = _ConversionWorker(cfg)
        self._worker.log_msg.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @pyqtSlot(str)
    def _append_log(self, msg: str):
        color_map = {"[OK]": "#34D399", "[ERROR]": "#EF4444",
                     "[WARNING]": "#FBBF24", "[INFO]": "#38BDF8"}
        color = "#94A3B8"
        for key, c in color_map.items():
            if msg.startswith(key):
                color = c
                break
        self.log_area.append(f'<span style="color:{color};">{msg}</span>')
        self.log_area.moveCursor(QTextCursor.MoveOperation.End)

    @pyqtSlot(object)
    def _on_finished(self, result: ConversionResult):
        self._progress.setVisible(False)
        self.btn_convert.setEnabled(True)
        self.btn_convert.setText("⚡  Dönüştür")

        if result.success:
            self._result_lbl.setText(
                f"✅  Başarılı! {result.output_path.name}  —  {result.elapsed_sec:.1f}s"
            )
            self._result_lbl.setStyleSheet(
                "color: #34D399; background:#001A0D; border:1px solid #34D399; "
                "border-radius:6px; padding: 8px; font-weight:600;"
            )
            self._status_bar.showMessage(f"✅ Tamamlandı: {result.output_path}")
            self._append_log(f"\n[OK] Toplam süre: {result.elapsed_sec:.1f} saniye")
        else:
            self._result_lbl.setText(f"❌  Hata: {result.message}")
            self._result_lbl.setStyleSheet(
                "color: #EF4444; background:#1A0000; border:1px solid #EF4444; "
                "border-radius:6px; padding: 8px;"
            )
            self._status_bar.showMessage(f"❌ Başarısız: {result.message}")
