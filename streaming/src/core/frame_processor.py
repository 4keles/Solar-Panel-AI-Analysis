import threading
import queue
import time
from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO
import cv2

from .exceptions import ProcessorNotInitializedError

class ThermalConverter:
    """RGB frame'i modelin beklediği termal/grayscale formatına dönüştürür."""
    COLORMAPS = {
        "grayscale": None,
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "magma": cv2.COLORMAP_MAGMA,
        "hot": cv2.COLORMAP_HOT,
    }

    def __init__(self, mode: str = "grayscale", alpha: float = 1.0, beta: int = 0):
        self.mode = mode.lower() if mode else "grayscale"
        self.alpha = alpha
        self.beta = beta

    def convert(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame
        try:
            if len(frame.shape) == 2:
                gray = frame
            elif frame.shape[2] == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            elif frame.shape[2] == 1:
                gray = frame.squeeze(2)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Termal kameralardan gelen 16-bit vb. formatlari 8-bit'e normalize et
            if gray.dtype != np.uint8:
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            if self.alpha != 1.0 or self.beta != 0:
                gray = cv2.convertScaleAbs(gray, alpha=self.alpha, beta=self.beta)
                
            if self.mode == "grayscale":
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            cmap = self.COLORMAPS.get(self.mode, cv2.COLORMAP_INFERNO)
            if cmap is None:
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            return cv2.applyColorMap(gray, cmap)
        except Exception:
            return frame


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # piksel, mutlak

@dataclass
class ProcessResult:
    frame: np.ndarray          # Orijinal kare — ASLA değiştirilmez
    detections: list[Detection]
    inference_ms: float        # Sadece inference süresi

class FrameProcessor:
    """Takes frames from a source queue, runs YOLO inference in a background thread, and pushes results to an output queue."""
    def __init__(self, model: YOLO, input_queue: queue.Queue, output_queue: queue.Queue,
                 conf: float = 0.25, iou: float = 0.45, device: str = "cpu",
                 thermal_mode: str | None = None, thermal_alpha: float = 1.0, thermal_beta: int = 0):
        self._model = model
        self.conf = conf
        self.iou = iou
        self.device = device
        
        # Thread & Queue architecture
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._is_running = False
        self._thread: threading.Thread | None = None

        self.image_alpha = thermal_alpha
        self.image_beta = thermal_beta
        self._thermal_converter = ThermalConverter(thermal_mode, 1.0, 0) if thermal_mode else None
        self.target_fps = 0  # 0 means unlimited
        self._last_raw_frame: np.ndarray | None = None
        self._force_reprocess: bool = False

    @property
    def thermal_mode(self) -> str | None:
        return self._thermal_converter.mode if self._thermal_converter else None

    @thermal_mode.setter
    def thermal_mode(self, mode: str | None) -> None:
        if mode:
            self._thermal_converter = ThermalConverter(mode, 1.0, 0)
        else:
            self._thermal_converter = None

    @property
    def thermal_alpha(self) -> float:
        return self.image_alpha

    @thermal_alpha.setter
    def thermal_alpha(self, val: float) -> None:
        self.image_alpha = val

    @property
    def thermal_beta(self) -> int:
        return self.image_beta

    @thermal_beta.setter
    def thermal_beta(self, val: int) -> None:
        self.image_beta = val


    def start(self) -> None:
        """İşleme iş parçacığını asenkron olarak başlatır."""
        if self._is_running:
            return
        if self._model is None:
            raise ProcessorNotInitializedError("FrameProcessor model yüklenmeden başlatılamaz.")
            
        self._is_running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """İşleme iş parçacığını durdurur."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def force_reprocess(self) -> None:
        """Forces the processor to re-run inference/thermal conversion on the last raw frame."""
        self._force_reprocess = True

    def _process_loop(self) -> None:
        """Sürekli olarak input_queue üzerinden okur ve infer edilen sonuçları output_queue'ya basar."""
        while self._is_running:
            loop_start = time.perf_counter()
            try:
                if self._force_reprocess and self._last_raw_frame is not None:
                    self._force_reprocess = False
                    frame = self._last_raw_frame.copy()
                    ret = True
                else:
                    ret, frame = self._input_queue.get(timeout=0.1)
                    if ret and frame is not None and frame.size > 0:
                        self._last_raw_frame = frame.copy()
                        
                if not ret or frame is None or frame.size == 0:
                    # Video EOF, pass the EOF signal nicely to output
                    self._put_to_output(ProcessResult(frame=np.zeros((0,0,3), dtype=np.uint8), detections=[], inference_ms=0))
                    continue
                    
                result = self.process(frame)
                self._put_to_output(result)
                
                # FPS Limiting
                if getattr(self, 'target_fps', 0) > 0:
                    target_time = 1.0 / self.target_fps
                    elapsed = time.perf_counter() - loop_start
                    if elapsed < target_time:
                        time.sleep(target_time - elapsed)
                        
            except queue.Empty:
                pass
            except Exception as e:
                # Log properly, don't crash the whole pipeline on a bad frame
                pass
                
    def _put_to_output(self, result: ProcessResult) -> None:
        # LIFO mechanism for output - if output is jammed, drop the oldest processed frame to keep UI fast
        if self._output_queue.full():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._output_queue.put(result, timeout=1.0)
        except queue.Full:
            pass

    def process(self, frame: np.ndarray) -> ProcessResult:
        """Tek bir frame için inference hesaplar."""
        if self._model is None:
            raise ProcessorNotInitializedError("Model yüklenmedi.")
            
        if frame.size == 0:
            return ProcessResult(frame=frame, detections=[], inference_ms=0.0)

        # Kontrast ve parlaklık uygula (Tüm modlarda RT etkisi)
        if self.image_alpha != 1.0 or self.image_beta != 0:
            # Sadece 8-bit ise convertScaleAbs çalışır, 16-bit ise önce 8-bit yapalım
            if frame.dtype != np.uint8:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame = cv2.convertScaleAbs(frame, alpha=self.image_alpha, beta=self.image_beta)

        # Termal dönüşüm (varsa) veya tek kanallı görüntüyü 3 kanala çevirme
        if self._thermal_converter:
            frame = self._thermal_converter.convert(frame)
        elif len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            if frame.dtype != np.uint8:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if len(frame.shape) == 3:
                frame = frame.squeeze(2)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            if frame.dtype != np.uint8:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        t_start = time.perf_counter()
        
        # GPU varsa 640, CPU'da 320 ile çok daha hızlı çalışır
        imgsz = 640 if self.device not in ("cpu",) else 320

        # prediction with object tracking (stabilizes bounding boxes across frames)
        results = self._model.track(
            source=frame,

            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=imgsz,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )
        
        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000.0

        detections = []
        if results and len(results) > 0:
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()
                names = self._model.names
                
                for box, cnf, cls in zip(boxes, confs, classes):
                    class_id = int(cls)
                    detections.append(Detection(
                        class_id=class_id,
                        class_name=names.get(class_id, f"class_{class_id}"),
                        confidence=float(cnf),
                        bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                    ))

        return ProcessResult(frame=frame, detections=detections, inference_ms=inference_ms)
