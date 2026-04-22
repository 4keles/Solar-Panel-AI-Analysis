import cv2
import numpy as np
from collections import defaultdict
from .frame_processor import ProcessResult

class Annotator:
    """Orijinal kareyi değiştirmeden kopyasına bounding-box ve HUD çizen bağımsız sınıf."""
    def __init__(self, class_colors: dict[int, tuple[int, int, int]], conf_threshold: float = 0.25):
        self.class_colors = class_colors
        self.conf_threshold = conf_threshold
        
        # Display settings from mock display config standard
        self.thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.text_color = (255, 255, 255)

    def draw(self, frame: np.ndarray, result: ProcessResult) -> np.ndarray:
        """Çizim yapılmış kareyi kopyalayarak geri döndürür."""
        annotated = frame.copy()
        
        if not result or not result.detections:
            return annotated
            
        for det in result.detections:
            if det.confidence < self.conf_threshold:
                continue
                
            x1, y1, x2, y2 = map(int, det.bbox_xyxy)
            color = self.class_colors.get(det.class_id, (255, 255, 255))
            
            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.thickness)
            
            # Draw Label
            label = f"{det.class_name} {det.confidence:.2f}"
            t_size = cv2.getTextSize(label, self.font, self.font_scale, 1)[0]
            
            # Background for label
            c2 = x1 + t_size[0] + 3
            c3 = y1 - t_size[1] - 4
            cv2.rectangle(annotated, (x1, y1), (c2, c3), color, -1)
            
            # Text
            cv2.putText(annotated, label, (x1 + 2, y1 - 2), self.font, self.font_scale, self.text_color, 1, cv2.LINE_AA)
            
        return annotated

    def draw_hud(self, frame: np.ndarray, fps: float, source_label: str, recording: bool = False) -> np.ndarray:
        """Kareye Head-Up Display nesnesi çizer (FPS, Source, Kayıt bilgileri vb). Orijinali DEĞİŞTİRMEZ."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Draw Background Overlay (Translucent)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (max(w, 250), 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

        # FPS & Source Label
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), self.font, self.font_scale, self.text_color, 2)
        cv2.putText(annotated, f"Source: {source_label}", (10, 60), self.font, self.font_scale, self.text_color, 2)
        
        if recording:
            # Draw recording indicator (Red Dot + REC)
            rec_x, rec_y = w - 120, 30
            cv2.circle(annotated, (rec_x, rec_y - 5), 8, (0, 0, 255), -1)
            cv2.putText(annotated, "REC", (rec_x + 15, rec_y), self.font, self.font_scale, (0, 0, 255), 2)
            
        return annotated
