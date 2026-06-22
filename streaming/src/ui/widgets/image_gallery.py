from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem
from PyQt6.QtCore import pyqtSignal, Qt
from pathlib import Path

class ImageGalleryWidget(QWidget):
    image_selected = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imageGalleryWidget")
        self.setStyleSheet("""
            QWidget#imageGalleryWidget {
                background-color: #0F1117;
                border-left: 1px solid #1E2A3A;
            }
        """)
        self.setMinimumWidth(250)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        self.lbl_title = QLabel("Resim Galerisi")
        self.lbl_title.setStyleSheet("color: #E2E8F0; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.lbl_title)
        
        self.lbl_info = QLabel("0 resim bulundu")
        self.lbl_info.setStyleSheet("color: #94A3B8; font-size: 11px;")
        layout.addWidget(self.lbl_info)
        
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #161B27;
                border: 1px solid #1E2A3A;
                border-radius: 4px;
                color: #CBD5E1;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #1E2A3A;
            }
            QListWidget::item:selected {
                background-color: #3B82F6;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #1E2A3A;
            }
        """)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self.list_widget)
        
        self._is_updating = False

    def set_images(self, images: list[Path]):
        self._is_updating = True
        self.list_widget.clear()
        
        total_size = 0
        for p in images:
            try:
                size_kb = p.stat().st_size / 1024
                total_size += size_kb
                text = f"{p.name}\n{size_kb:.1f} KB"
            except Exception:
                text = p.name
            
            item = QListWidgetItem(text)
            self.list_widget.addItem(item)
            
        self.lbl_info.setText(f"{len(images)} resim, Toplam: {total_size/1024:.1f} MB")
        self._is_updating = False
        
    def select_image(self, index: int):
        self._is_updating = True
        if 0 <= index < self.list_widget.count():
            self.list_widget.setCurrentRow(index)
            self.list_widget.scrollToItem(self.list_widget.item(index))
        self._is_updating = False

    def _on_row_changed(self, row: int):
        if not self._is_updating and row >= 0:
            self.image_selected.emit(row)
