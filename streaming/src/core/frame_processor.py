import threading
import queue
import time
from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO

from .exceptions import ProcessorNotInitializedError

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
    def __init__(self, model: YOLO, input_queue: queue.Queue, output_queue: queue.Queue, conf: float = 0.25, iou: float = 0.45, device: str = "cpu"):
        self._model = model
        self.conf = conf
        self.iou = iou
        self.device = device
        
        # Thread & Queue architecture
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._is_running = False
        self._thread: threading.Thread | None = None

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

    def _process_loop(self) -> None:
        """Sürekli olarak input_queue üzerinden okur ve infer edilen sonuçları output_queue'ya basar."""
        while self._is_running:
            try:
                ret, frame = self._input_queue.get(timeout=0.1)
                if not ret or frame is None:
                    # Video EOF, pass the EOF signal nicely to output
                    self._put_to_output(ProcessResult(frame=np.zeros(0), detections=[], inference_ms=0))
                    continue
                    
                result = self.process(frame)
                self._put_to_output(result)
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
            
        if frame.size == 0 or len(frame.shape) != 3:
            return ProcessResult(frame=frame, detections=[], inference_ms=0.0)

        t_start = time.perf_counter()
        
        # prediction with raw frame
        results = self._model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
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
