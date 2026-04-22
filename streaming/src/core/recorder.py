import cv2
import numpy as np
import threading
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from .exceptions import RecorderAlreadyRunningError, RecorderNotRunningError, RecorderSetupError

@dataclass
class RecordingSummary:
    output_path: Path
    frame_count: int
    duration_sec: float
    file_size_bytes: int

class VideoRecorder:
    """Belirtilen dizine OpenCV üzerinden asenkron / multi-thread safe video yazıcı."""
    
    def __init__(self, output_dir: Path, fps: float, resolution: tuple[int, int], codec: str = "mp4v"):
        self._output_dir = output_dir
        self._fps = fps
        self._resolution = resolution
        self._codec = codec
        
        self._writer: cv2.VideoWriter | None = None
        self._recording: bool = False
        self._lock = threading.Lock()
        
        self._current_file: Path | None = None
        self._frame_count: int = 0

    def start(self, filename: str | None = None) -> Path:
        """Yeni bir kayıt başlatır ve hedeflenen dosya yolunu döner."""
        with self._lock:
            if self._recording:
                raise RecorderAlreadyRunningError("Kayıt halihazırda devam ediyor.")
                
            try:
                self._output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RecorderSetupError(f"Çıktı klasörü oluşturulamadı: {self._output_dir} ({e})")
                
            if not filename:
                filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            if not filename.endswith(".mp4"):
                filename += ".mp4"
                
            output_path = self._output_dir / filename
            
            fourcc = cv2.VideoWriter_fourcc(*self._codec)
            self._writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self._fps,
                (self._resolution[0], self._resolution[1])
            )
            
            if not self._writer.isOpened():
                raise RecorderSetupError("VideoWriter başlatılamadı. Codec eksik olabilir.")
                
            self._recording = True
            self._frame_count = 0
            self._current_file = output_path
            
            return output_path

    def write(self, frame: np.ndarray) -> None:
        """Sıradaki kareyi kaydeder. Non-blocking/thread-safe lock ile."""
        with self._lock:
            if self._recording and self._writer is not None:
                self._writer.write(frame)
                self._frame_count += 1

    def stop(self) -> RecordingSummary:
        """Kaydı sonlandırır ve özetini döner."""
        with self._lock:
            if not self._recording or self._writer is None or self._current_file is None:
                raise RecorderNotRunningError("Durdurulacak aktif bir kayıt bulunamadı.")
                
            self._writer.release()
            self._writer = None
            self._recording = False
            
            size = 0
            if self._current_file.exists():
                size = self._current_file.stat().st_size
                
            duration = self._frame_count / self._fps if self._fps > 0 else 0.0
            
            summary = RecordingSummary(
                output_path=self._current_file,
                frame_count=self._frame_count,
                duration_sec=duration,
                file_size_bytes=size
            )
            
            self._current_file = None
            return summary

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording
