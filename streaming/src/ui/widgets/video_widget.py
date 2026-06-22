"""VideoWidget — Qt6 frame display widget.

Renders BGR numpy frames onto a QLabel while preserving aspect ratio.
Uses QImage directly (no Pillow dependency) for maximum performance.
"""

import numpy as np
from PyQt6.QtWidgets import QLabel, QSizePolicy, QPushButton
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen
from PyQt6.QtCore import Qt, QSize, pyqtSignal
import cv2


class VideoWidget(QLabel):
    """High-performance video frame display widget.
    
    Accepts BGR numpy arrays and renders them as Qt pixmaps.
    Maintains aspect ratio and shows a placeholder when idle.
    """

    PLACEHOLDER_BG = QColor(10, 15, 25)
    PLACEHOLDER_TEXT_COLOR = QColor(55, 65, 81)
    
    prev_clicked = pyqtSignal()
    next_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("videoWidget")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #0A0F19; border-radius: 0px;")
        
        self._current_pixmap: QPixmap | None = None
        self._is_image_mode = False
        self._image_info = ""
        
        # Overlay buttons for Image Mode
        self.btn_prev = QPushButton("◀", self)
        self.btn_prev.setFixedSize(40, 60)
        self.btn_prev.setStyleSheet("QPushButton { background-color: rgba(15, 23, 42, 0.6); color: white; border: none; border-radius: 4px; font-size: 24px; } QPushButton:hover { background-color: rgba(56, 189, 248, 0.8); }")
        self.btn_prev.clicked.connect(self.prev_clicked.emit)
        self.btn_prev.hide()
        
        self.btn_next = QPushButton("▶", self)
        self.btn_next.setFixedSize(40, 60)
        self.btn_next.setStyleSheet("QPushButton { background-color: rgba(15, 23, 42, 0.6); color: white; border: none; border-radius: 4px; font-size: 24px; } QPushButton:hover { background-color: rgba(56, 189, 248, 0.8); }")
        self.btn_next.clicked.connect(self.next_clicked.emit)
        self.btn_next.hide()
        
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        """Draw a dark placeholder with icon when no stream is active."""
        size = self.size() if self.width() > 0 else QSize(640, 480)
        placeholder = QPixmap(size)
        placeholder.fill(self.PLACEHOLDER_BG)

        painter = QPainter(placeholder)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Center icon area
        cx, cy = size.width() // 2, size.height() // 2

        # Draw camera icon (simple geometric)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(30, 41, 59))
        painter.drawRoundedRect(cx - 60, cy - 40, 120, 80, 10, 10)
        
        painter.setBrush(QColor(51, 65, 85))
        painter.drawEllipse(cx - 20, cy - 20, 40, 40)
        painter.setBrush(QColor(15, 23, 42))
        painter.drawEllipse(cx - 12, cy - 12, 24, 24)
        
        # Lens highlight
        painter.setBrush(QColor(100, 116, 139, 80))
        painter.drawEllipse(cx - 6, cy - 14, 8, 6)

        # Text
        font = QFont("Inter", 12)
        painter.setFont(font)
        painter.setPen(self.PLACEHOLDER_TEXT_COLOR)
        painter.drawText(
            0, cy + 60, size.width(), 30,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "Kaynak seçip  ▶ Başlat  butonuna basın"
        )
        
        font_small = QFont("Inter", 9)
        painter.setFont(font_small)
        painter.setPen(QColor(30, 41, 59))
        painter.drawText(
            0, cy + 88, size.width(), 20,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "Kamera  •  Video Dosyası  •  RTSP Akışı"
        )

        painter.end()
        self.setPixmap(placeholder)

    def set_image_mode(self, enabled: bool) -> None:
        self._is_image_mode = enabled
        if enabled:
            self.btn_prev.show()
            self.btn_next.show()
            self._update_button_positions()
        else:
            self.btn_prev.hide()
            self.btn_next.hide()
            self._image_info = ""
        self.update()

    def set_image_info(self, info: str) -> None:
        self._image_info = info
        self.update()

    def _update_button_positions(self):
        # Center vertically, place on edges
        y = (self.height() - self.btn_prev.height()) // 2
        self.btn_prev.move(10, y)
        self.btn_next.move(self.width() - self.btn_next.width() - 10, y)

    def render_frame(self, frame: np.ndarray) -> None:
        """Convert a BGR numpy array to QPixmap and display it.
        
        Args:
            frame: BGR image array of shape (H, W, 3) uint8.
        """
        if frame is None or frame.size == 0:
            return

        # Ensure frame is 3 channel 8-bit for QImage
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            if frame.dtype != np.uint8:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if len(frame.shape) == 3:
                frame = frame.squeeze(2)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            if frame.dtype != np.uint8:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            if frame.dtype != np.uint8:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame = frame[:, :, ::-1].copy() # BGR -> RGB
            
        h, w, ch = frame.shape
        if ch != 3:
            return
        
        q_img = QImage(
            frame.data,
            w, h,
            w * 3,
            QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(q_img)
        self._current_pixmap = pixmap
        
        # Scale to widget size maintaining aspect ratio
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def clear_frame(self) -> None:
        """Reset to placeholder state."""
        self._current_pixmap = None
        self._show_placeholder()

    def resizeEvent(self, event):
        """Re-scale current frame on widget resize."""
        super().resizeEvent(event)
        self._update_button_positions()
        if self._current_pixmap:
            scaled = self._current_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
        else:
            self._show_placeholder()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._is_image_mode and self._image_info and self._current_pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Semi-transparent background for text
            painter.setBrush(QColor(15, 23, 42, 200))
            painter.setPen(Qt.PenStyle.NoPen)
            
            font = QFont("Inter", 11, QFont.Weight.Bold)
            painter.setFont(font)
            fm = painter.fontMetrics()
            
            lines = self._image_info.split('\n')
            max_width = max([fm.horizontalAdvance(line) for line in lines]) if lines else 0
            text_height = fm.height() * len(lines)
            
            padding = 10
            rect_w = max_width + padding * 2
            rect_h = text_height + padding * 2
            
            x = 20
            y = self.height() - rect_h - 20
            
            painter.drawRoundedRect(x, y, rect_w, rect_h, 6, 6)
            
            painter.setPen(QColor(255, 255, 255))
            current_y = y + padding + fm.ascent()
            for line in lines:
                painter.drawText(x + padding, current_y, line)
                current_y += fm.height()
                
            painter.end()
