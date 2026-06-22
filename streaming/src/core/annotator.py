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
            cv2.putText(annotated, label, (x1 + 2, y1 - 2), self.font, self.font_scale, (0, 0, 0), 1, cv2.LINE_AA)
            
        return annotated

    def draw_hud(self, frame: np.ndarray, fps: float, source_label: str, recording: bool = False,
                 inference_ms: float | None = None, conn_status: str | None = None,
                 thermal_mode: str | None = None) -> np.ndarray:
        """Kareye Head-Up Display nesnesi çizer (FPS, Source, Kayıt, Gecikme, Termal bilgiler vb). Orijinali DEĞİŞTİRMEZ."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Üst şerit yarı saydam arka plan (0,0) -> (w, 35)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 35), (10, 15, 20), -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

        # Yazı ayarları
        y_offset = 22
        x_offset = 12

        # 1. Bağlantı Durumu (Gösterge Noktası + Metin)
        dot_color = (128, 128, 128)  # Disconnected / default
        status_text = "DISCONNECTED"
        if conn_status:
            c_str = str(conn_status).lower()
            if "connected" in c_str:
                dot_color = (0, 220, 0)      # Neon Green BGR
                status_text = "CONNECTED"
            elif "reconnecting" in c_str:
                dot_color = (0, 200, 255)    # Gold/Yellow BGR
                status_text = "RECONNECTING"
            elif "failed" in c_str:
                dot_color = (0, 0, 255)      # Red BGR
                status_text = "CONN FAILED"

        cv2.circle(annotated, (x_offset + 6, y_offset - 5), 5, dot_color, -1)
        cv2.putText(annotated, status_text, (x_offset + 18, y_offset), self.font, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        x_offset += 140

        # 2. Kaynak Adı (Kısaltılmış)
        src_display = source_label[:20] + "..." if len(source_label) > 20 else source_label
        cv2.putText(annotated, f"SRC: {src_display}", (x_offset, y_offset), self.font, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        x_offset += 190

        # 3. FPS Bilgisi
        cv2.putText(annotated, f"FPS: {fps:.1f}", (x_offset, y_offset), self.font, 0.4, (56, 189, 248), 1, cv2.LINE_AA)
        x_offset += 90

        # 4. Gecikme (Inference Time)
        if inference_ms is not None:
            cv2.putText(annotated, f"INF: {inference_ms:.1f}ms", (x_offset, y_offset), self.font, 0.4, (167, 139, 250), 1, cv2.LINE_AA)
            x_offset += 100

        # 5. Termal Badge (Sağ Tarafta)
        if thermal_mode:
            badge_text = f"THERMAL: {thermal_mode.upper()}"
            badge_size = cv2.getTextSize(badge_text, self.font, 0.35, 1)[0]
            bx1 = w - badge_size[0] - 20
            if recording:
                bx1 -= 80
            
            cv2.rectangle(annotated, (bx1, 6), (bx1 + badge_size[0] + 10, 28), (14, 116, 144), -1) # Teal BGR
            cv2.putText(annotated, badge_text, (bx1 + 5, y_offset - 2), self.font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # 6. Kayıt Işığı (Sağ En Uçta)
        if recording:
            rec_x = w - 65
            cv2.circle(annotated, (rec_x, y_offset - 5), 5, (0, 0, 255), -1)
            cv2.putText(annotated, "REC", (rec_x + 10, y_offset), self.font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        return annotated

