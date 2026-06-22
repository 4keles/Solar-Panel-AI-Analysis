import threading
import queue
import time
from pathlib import Path
from enum import Enum, auto
from typing import Callable
import numpy as np

import cv2

from .exceptions import SourceNotOpenError, SourceOpenError, ConnectionStatus

class SourceType(Enum):
    CAMERA = auto()
    VIDEO = auto()
    RTSP = auto()
    IP = auto()
    RTMP = auto()
    IMAGE_DIR = auto()

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
        self._is_paused = False
        self._playback_speed = 1.0
        self._cap_lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._queue: queue.Queue[tuple[bool, np.ndarray | None]] = queue.Queue(maxsize=max_queue_size)

    def _detect_source_type(self) -> SourceType:
        src_str = str(self._raw_source).lower()
        if isinstance(self._raw_source, int) or src_str.isdigit():
            return SourceType.CAMERA
        if src_str.startswith("rtsp://"):
            return SourceType.RTSP
        if src_str.startswith("rtmp://"):
            return SourceType.RTMP
        if src_str.startswith(("http://", "https://")):
            return SourceType.IP
        p = Path(str(self._raw_source))
        if p.is_dir():
            return SourceType.IMAGE_DIR
        return SourceType.VIDEO

    def get_queue(self) -> queue.Queue:
        """Kuyruğa güvenli erişim sağlayan public API."""
        return self._queue

    def open(self) -> None:
        """Video kaynağını başlatır ve yakalama iş parçacığını (thread) devreye sokar."""
        if self._is_running:
            return

        source_val = self._raw_source
        if self._type == SourceType.CAMERA and isinstance(source_val, str):
            source_val = int(source_val)
        elif self._type in (SourceType.VIDEO, SourceType.RTSP, SourceType.IP, SourceType.RTMP):
            source_val = str(source_val)

        # OpenCV capture initialization
        # Use cv2.CAP_V4L2 for camera if on linux, or let OpenCV guess.
        self._cap = cv2.VideoCapture(source_val)

        if not self._cap.isOpened():
            raise SourceOpenError(f"Video kaynağı açılamadı: {self._raw_source}")

        # Try to request MJPG format (some cameras limit to 30 FPS on YUYV)
        if self._type == SourceType.CAMERA:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # RTSP/IP/RTMP optimization
        if self._type in (SourceType.RTSP, SourceType.IP, SourceType.RTMP):
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
        # For video files, we want to play at the original speed
        frame_time = 1.0 / self._fps if self._fps > 0 else 0.033
        
        while self._is_running and self._cap is not None:
            if self._is_paused:
                time.sleep(0.05)
                continue
                
            loop_start = time.perf_counter()
            with self._cap_lock:
                if self._cap is None:
                    break
                ret, frame = self._cap.read()
            
            # Queue sınırına ulaşıldıysa eski frame'i at (LIFO / Drop Frame)
            # Bu sayede inference yetişemese bile UI canlı akışı kaybetmez.
            if self._queue.full() and self.is_live():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._queue.put((ret, frame), timeout=1.0)
            except queue.Full:
                pass

            if not ret:
                if self.is_file():
                    self._is_paused = True
                    time.sleep(0.05)
                    continue
                else:
                    break  # EOF or error for live streams

            if self.is_file():
                elapsed = time.perf_counter() - loop_start
                sleep_time = (frame_time / self._playback_speed) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
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
            
        with self._cap_lock:
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
        return self._type in (SourceType.CAMERA, SourceType.RTSP, SourceType.IP, SourceType.RTMP)

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False

    def set_playback_speed(self, speed: float) -> None:
        self._playback_speed = speed

    def seek(self, frame_idx: int) -> None:
        with self._cap_lock:
            if self._cap is not None and self.is_file():
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                # Clear the queue so the new frames are immediately available
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break

    def get_position(self) -> int:
        with self._cap_lock:
            if self._cap is not None and self.is_file():
                return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        return 0

    def get_total_frames(self) -> int:
        with self._cap_lock:
            if self._cap is not None and self.is_file():
                return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class ResilientVideoSource(VideoSource):
    """Canlı kaynaklarda bağlantı kaybını tolere edebilen ve reconnect mekanizması içeren kaynak okuyucu."""

    def __init__(self, source: str | int | Path, max_queue_size: int = 2, max_retries: int = 10, retry_interval_sec: float = 3.0):
        super().__init__(source, max_queue_size)
        self.max_retries = max_retries
        self.retry_interval_sec = retry_interval_sec
        self._connection_status = ConnectionStatus.DISCONNECTED
        self._status_callback: Callable[[ConnectionStatus], None] | None = None

    def set_status_callback(self, cb: Callable[[ConnectionStatus], None]) -> None:
        self._status_callback = cb
        if cb:
            cb(self._connection_status)

    def _update_status(self, status: ConnectionStatus) -> None:
        if self._connection_status != status:
            self._connection_status = status
            if self._status_callback:
                try:
                    self._status_callback(status)
                except Exception:
                    pass

    def open(self) -> None:
        """Video kaynağını başlatır ve gerekirse arka planda reconnect modunda çalıştırır."""
        if self._is_running:
            return

        source_val = self._raw_source
        if self._type == SourceType.CAMERA and isinstance(source_val, str):
            source_val = int(source_val)
        elif self._type in (SourceType.VIDEO, SourceType.RTSP, SourceType.IP, SourceType.RTMP):
            source_val = str(source_val)

        self._cap = cv2.VideoCapture(source_val)

        if self._type == SourceType.CAMERA:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        elif self._type in (SourceType.RTSP, SourceType.IP, SourceType.RTMP):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            if self.is_live():
                self._update_status(ConnectionStatus.RECONNECTING)
                self._is_running = True
                self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._capture_thread.start()
                return
            else:
                raise SourceOpenError(f"Video dosyası açılamadı: {self._raw_source}")

        # Meta properties
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self._fps = float(fps)
        self._resolution = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        self._update_status(ConnectionStatus.CONNECTED)
        self._is_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """Kare yakalayan arka plan döngüsü. Bağlantı kesilirse otomatik reconnect döngüsüne girer."""
        source_val = self._raw_source
        if self._type == SourceType.CAMERA and isinstance(source_val, str):
            source_val = int(source_val)
        elif self._type in (SourceType.VIDEO, SourceType.RTSP, SourceType.IP, SourceType.RTMP):
            source_val = str(source_val)

        retries = 0
        while self._is_running:
            if self._is_paused:
                time.sleep(0.05)
                continue
                
            if self._cap is None or not self._cap.isOpened():
                self._update_status(ConnectionStatus.RECONNECTING)
                with self._cap_lock:
                    if self._cap is not None:
                        self._cap.release()
                        self._cap = None
                
                # Reconnection attempts logic
                success = False
                for attempt in range(self.max_retries):
                    if not self._is_running:
                        break
                    try:
                        with self._cap_lock:
                            self._cap = cv2.VideoCapture(str(self._raw_source))
                            if self._cap.isOpened():
                                if self._type in (SourceType.RTSP, SourceType.IP, SourceType.RTMP):
                                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                success = True
                                break
                    except Exception:
                        pass
                    time.sleep(self.retry_interval_sec)

                if not success:
                    self._update_status(ConnectionStatus.ERROR)
                    break
                else:
                    self._update_status(ConnectionStatus.CONNECTED)
                    with self._cap_lock:
                        if self._cap is not None:
                            fps = self._cap.get(cv2.CAP_PROP_FPS)
                            if fps > 0:
                                self._fps = float(fps)
                            self._resolution = (
                                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            )
                continue

            loop_start = time.perf_counter()
            with self._cap_lock:
                if self._cap is None:
                    break
                ret, frame = self._cap.read()
            if not ret or frame is None:
                if self.is_live():
                    if self._cap is not None:
                        try:
                            self._cap.release()
                        except Exception:
                            pass
                        self._cap = None
                    self._update_status(ConnectionStatus.RECONNECTING)
                    continue
                else:
                    # Video EOF, signal EOF and break
                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        self._queue.put((False, None), timeout=1.0)
                    except queue.Full:
                        pass
                    break

            retries = 0
            self._update_status(ConnectionStatus.CONNECTED)

            # Queue drop-frame logic (LIFO) for live streams
            if self._queue.full() and self.is_live():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._queue.put((ret, frame), timeout=1.0)
            except queue.Full:
                pass

            if self.is_file():
                frame_time = 1.0 / self._fps if self._fps > 0 else 0.033
                elapsed = time.perf_counter() - loop_start
                sleep_time = (frame_time / self._playback_speed) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.001)

        self._update_status(ConnectionStatus.DISCONNECTED)


class ImageDirectorySource(VideoSource):
    """Kaynak olarak bir klasör verildiğinde içindeki resimleri oynatan sınıf."""

    def __init__(self, source: str | Path, max_queue_size: int = 2):
        super().__init__(source, max_queue_size)
        self._images: list[Path] = []
        self._idx = 0
        self._fps = 5.0 # Default 5 FPS for image playback

    def open(self) -> None:
        if self._is_running:
            return
            
        p = Path(str(self._raw_source))
        if not p.is_dir():
            raise SourceOpenError(f"Geçersiz resim klasörü: {self._raw_source}")
            
        # Collect all valid images
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp"):
            self._images.extend(p.rglob(ext))
            self._images.extend(p.rglob(ext.upper()))
            
        # Natural sorting or alphabetical
        self._images = sorted(list(set(self._images)))
        
        if not self._images:
            raise SourceOpenError(f"Klasörde resim bulunamadı: {self._raw_source}")

        # Get resolution from the first image
        first_img = cv2.imread(str(self._images[0]), cv2.IMREAD_UNCHANGED)
        if first_img is not None:
            h, w = first_img.shape[:2]
            self._resolution = (w, h)
            
        self._idx = 0
        self._is_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        while self._is_running:
            if self._is_paused:
                time.sleep(0.05)
                continue
                
            if self._idx >= len(self._images):
                # End of folder
                self._is_paused = True
                time.sleep(0.05)
                continue
                
            loop_start = time.perf_counter()
            img_path = str(self._images[self._idx])
            frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            # Fallback for TIFFs or unsupported formats
            if frame is None:
                try:
                    import imageio.v3 as iio
                    frame = iio.imread(img_path)
                    # Convert to BGR if it's RGB
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    elif len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
                except Exception as e:
                    pass
            
            if frame is None:
                try:
                    from PIL import Image
                    with Image.open(img_path) as pil_img:
                        frame = np.array(pil_img)
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    elif len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
                except Exception as e:
                    pass

            if frame is None:
                # Log or print so we know it failed
                print(f"FAILED to read image: {img_path}")
                self._idx += 1
                continue
                
            try:
                self._queue.put((True, frame), timeout=1.0)
                self._idx += 1
            except queue.Full:
                pass
                
            frame_time = 1.0 / self._fps if self._fps > 0 else 1.0
            elapsed = time.perf_counter() - loop_start
            sleep_time = (frame_time / self._playback_speed) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def release(self) -> None:
        self._is_running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
            
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def is_file(self) -> bool:
        return True # Treat as a video file so playback controls work
        
    def is_live(self) -> bool:
        return False
        
    def is_image_dir(self) -> bool:
        return True
        
    def set_fps(self, fps: float) -> None:
        if fps > 0:
            self._fps = fps
            
    def seek(self, frame_idx: int) -> None:
        if 0 <= frame_idx < len(self._images):
            self._idx = frame_idx
            # Clear queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                    
            # If paused, immediately push the frame to queue so UI updates
            if self._is_paused:
                frame = cv2.imread(str(self._images[self._idx]), cv2.IMREAD_UNCHANGED)
                if frame is not None:
                    try:
                        self._queue.put((True, frame), timeout=0.1)
                    except queue.Full:
                        pass
                        
    def get_position(self) -> int:
        return self._idx
        
    def get_total_frames(self) -> int:
        return len(self._images)

    def get_image_list(self):
        return self._images

    def get_current_image_details(self) -> str:
        if not self._images:
            return ""
        idx = min(self._idx, len(self._images) - 1)
        p = self._images[idx]
        try:
            size_kb = p.stat().st_size / 1024
        except Exception:
            size_kb = 0.0
        return f"Dosya: {p.name}\nBoyut: {size_kb:.1f} KB\nÇözünürlük: {self._resolution[0]}x{self._resolution[1]}"

