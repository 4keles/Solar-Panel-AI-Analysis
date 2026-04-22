import threading
import queue
import time
from pathlib import Path
from enum import Enum, auto
import numpy as np

import cv2

from .exceptions import SourceNotOpenError, SourceOpenError

class SourceType(Enum):
    CAMERA = auto()
    VIDEO = auto()
    RTSP = auto()

class VideoSource:
    """YOLO ve Streaming için Thread-Safe I/O destekli kaynak okuyucu."""

    def __init__(self, source: str | int | Path, max_queue_size: int = 2):
        self._raw_source = source
        self._type = self._detect_source_type()
        
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0
        self._resolution: tuple[int, int] = (640, 480)
        
        # Multithreading / Queue
        self._is_running = False
        self._capture_thread: threading.Thread | None = None
        self._queue: queue.Queue[tuple[bool, np.ndarray | None]] = queue.Queue(maxsize=max_queue_size)

    def _detect_source_type(self) -> SourceType:
        if isinstance(self._raw_source, int) or (isinstance(self._raw_source, str) and self._raw_source.isdigit()):
            return SourceType.CAMERA
        if str(self._raw_source).startswith(("rtsp://", "http://", "https://")):
            return SourceType.RTSP
        return SourceType.VIDEO

    def open(self) -> None:
        """Video kaynağını başlatır ve yakalama iş parçacığını (thread) devreye sokar."""
        if self._is_running:
            return

        source_val = self._raw_source
        if self._type == SourceType.CAMERA and isinstance(source_val, str):
            source_val = int(source_val)
        elif self._type == SourceType.VIDEO:
            source_val = str(source_val)

        # OpenCV capture initialization
        # Use cv2.CAP_V4L2 for camera if on linux, or let OpenCV guess.
        self._cap = cv2.VideoCapture(source_val)

        if not self._cap.isOpened():
            raise SourceOpenError(f"Video kaynağı açılamadı: {self._raw_source}")

        # RTSP optimization
        if self._type == SourceType.RTSP:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Meta properties
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self._fps = float(fps)
        self._resolution = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        # Start capture thread
        self._is_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """Arka plan thread'i. Frame yakalayıp LIFO mantığı ile queue'ya aktarır."""
        while self._is_running and self._cap is not None:
            ret, frame = self._cap.read()
            
            # Queue sınırına ulaşıldıysa eski frame'i at (LIFO / Drop Frame)
            # Bu sayede inference yetişemese bile UI canlı akışı kaybetmez.
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._queue.put((ret, frame), timeout=1.0)
            except queue.Full:
                pass

            if not ret:
                break  # EOF or error

            # Yield thread
            time.sleep(0.001)

    def read(self) -> tuple[bool, np.ndarray]:
        """Açık kaynaktan sıradaki Frame'i çeker."""
        if not self._is_running or self._cap is None:
            raise SourceNotOpenError("Kaynağı okumadan önce open() çağrılmalı.")
        
        try:
            return self._queue.get(timeout=2.0)
        except queue.Empty:
            # 2s içinde frame gelmezse kaynak tükenmiş veya takılmış sayarız.
            return False, np.zeros((0,0,3), dtype=np.uint8)

    def release(self) -> None:
        """Kaynağı serbest bırakır."""
        self._is_running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
            
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def get_fps(self) -> float:
        return self._fps

    def get_resolution(self) -> tuple[int, int]:
        return self._resolution

    def is_file(self) -> bool:
        return self._type == SourceType.VIDEO

    def is_live(self) -> bool:
        return self._type in (SourceType.CAMERA, SourceType.RTSP)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
