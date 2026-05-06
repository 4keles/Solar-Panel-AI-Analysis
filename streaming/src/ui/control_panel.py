import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import queue
import threading
import sys
from pathlib import Path

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
    _DEVICE = "cuda:0" if _CUDA_AVAILABLE else "cpu"
    _DEVICE_LABEL = f"GPU: {torch.cuda.get_device_name(0)}" if _CUDA_AVAILABLE else "CPU (CUDA yok)"
except Exception:
    _CUDA_AVAILABLE = False
    _DEVICE = "cpu"
    _DEVICE_LABEL = "CPU"

# Fix sys import to use logger properly
_scripts_dir = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from utils.logger import get_logger

from ..core.model_loader import ModelLoader
from ..core.source_manager import VideoSource
from ..core.frame_processor import FrameProcessor
from ..core.annotator import Annotator
from ..core.recorder import VideoRecorder
from ..utils.class_colors import generate_class_colors
from ..utils.fps_counter import FPSCounter

logger = get_logger(__name__)

class ControlPanel:
    def __init__(self, root: tk.Tk, config: dict):
        self.root = root
        self.config = config
        
        self.root.title(config.get("display", {}).get("window_title", "Solar Panel OD"))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Modules
        self.model_loader = ModelLoader()
        self._source: VideoSource | None = None
        self._processor: FrameProcessor | None = None
        self._annotator: Annotator | None = None
        self._recorder: VideoRecorder | None = None
        
        # Queues
        self.q_in = queue.Queue(maxsize=3)
        self.q_out = queue.Queue(maxsize=3) # UI queue
        
        # UI State
        self.is_running = False
        self._fps_counter = FPSCounter()
        
        # UI Layout
        self._setup_ui()
        self._populate_models()
        
        # Start core loop if auto-starting based on config
        self._schedule_update()

    def _setup_ui(self):
        # Top Frame for controls
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Source string
        ttk.Label(control_frame, text="Source:").grid(row=0, column=0, padx=5)
        self.source_var = tk.StringVar(value=str(self.config.get("source", {}).get("camera_id", 0)))
        ttk.Entry(control_frame, textvariable=self.source_var).grid(row=0, column=1, padx=5)
        
        # Model selection
        ttk.Label(control_frame, text="Model:").grid(row=0, column=2, padx=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=0, column=3, padx=5)
        
        # Confidence slider
        ttk.Label(control_frame, text="Conf:").grid(row=0, column=4, padx=5)
        self.conf_var = tk.DoubleVar(value=self.config.get("model", {}).get("conf", 0.25))
        ttk.Scale(control_frame, variable=self.conf_var, from_=0.0, to=1.0, command=self._on_conf_change).grid(row=0, column=5)
        
        # Buttons
        self.btn_toggle = ttk.Button(control_frame, text="▶ Start", command=self.toggle_stream)
        self.btn_toggle.grid(row=0, column=6, padx=5)
        
        self.btn_rec = ttk.Button(control_frame, text="⏺ Record", command=self.toggle_record, state=tk.DISABLED)
        self.btn_rec.grid(row=0, column=7, padx=5)

        # Device label
        device_color = "#00cc44" if _CUDA_AVAILABLE else "#ff8800"
        self.device_label = tk.Label(control_frame, text=f"⚡ {_DEVICE_LABEL}", fg=device_color, font=("Helvetica", 9, "bold"))
        self.device_label.grid(row=0, column=8, padx=10)
        
        # Video Canvas
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", width=640, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _populate_models(self):
        models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models"
        pt_models = list(models_dir.rglob("*.pt"))
        onnx_models = list(models_dir.rglob("*.onnx"))
        engine_models = list(models_dir.rglob("*.engine"))
        self._available_models = pt_models + onnx_models + engine_models
        if self._available_models:
            # Show relative path from models/ to make versions visible (e.g. v1.0.3/best.pt)
            self.model_combo["values"] = [str(p.relative_to(models_dir)) for p in self._available_models]
            self.model_combo.current(0)
            
            # Select target model if config has one
            config_mod = self.config.get("model", {}).get("path", "")
            if config_mod:
                config_mod_name = Path(config_mod).name
                for i, p in enumerate(self._available_models):
                    if p.name == config_mod_name:
                        self.model_combo.current(i)
                        break

    def _on_conf_change(self, *args):
        if self._processor:
            self._processor.conf = float(self.conf_var.get())

    def toggle_stream(self):
        if self.is_running:
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def toggle_record(self):
        if not self._recorder:
            return
            
        if self._recorder.is_recording():
            summary = self._recorder.stop()
            self.btn_rec.config(text="⏺ Record")
            logger.info("recording_stopped", summary=vars(summary))
        else:
            try:
                self._recorder.start()
                self.btn_rec.config(text="■ Stop Rec")
                logger.info("recording_started")
            except Exception as e:
                logger.error("recorder_error", error=str(e))

    def _start_pipeline(self):
        self.btn_toggle.config(text="⌛ Loading...")
        self.root.update()
        
        # 1. Load Model
        try:
            model_path_str = self.model_combo.get()
            model_path = next((p for p in self._available_models if p.name == model_path_str), None)
            
            if model_path is None and self.config.get("model", {}).get("path"):
                model_path = Path(self.config["model"]["path"])
                
            if model_path:
                model = self.model_loader.load(model_path)
            else:
                model = self.model_loader.load_latest(Path(__file__).resolve().parent.parent.parent.parent / "models")
                
            class_names = self.model_loader.get_class_names()
        except Exception as e:
            logger.error("pipeline_error", stage="model", error=str(e))
            self.btn_toggle.config(text="▶ Start")
            return

        # 2. Setup Source
        try:
            raw_src = self.source_var.get()
            if raw_src.isdigit():
                raw_src = int(raw_src)
            self._source = VideoSource(source=raw_src, max_queue_size=3)
            self._source.open()
        except Exception as e:
            logger.error("pipeline_error", stage="source", error=str(e))
            self.btn_toggle.config(text="▶ Start")
            return

        # 3. Setup Annotator & Processor & Recorder
        colors = generate_class_colors(class_names)
        self._annotator = Annotator(class_colors=colors, conf_threshold=self.conf_var.get())
        
        # GPU varsa cuda:0, yoksa cpu — config'e bakmadan otomatik seç
        device = _DEVICE
        logger.info("inference_device", device=device, cuda=_CUDA_AVAILABLE)

        self._processor = FrameProcessor(
            model=model,
            input_queue=self._source._queue,
            output_queue=self.q_out,
            conf=float(self.conf_var.get()),
            device=device
        )
        self._processor.start()
        
        self._recorder = VideoRecorder(
            output_dir=Path(self.config.get("recording", {}).get("output_dir", "output/recordings")),
            fps=self._source.get_fps(),
            resolution=self._source.get_resolution()
        )
        
        self.is_running = True
        self.btn_toggle.config(text="■ Stop")
        self.btn_rec.config(state=tk.NORMAL)
        
        # Handle config recording start
        if self.config.get("recording", {}).get("enabled", False):
            self.toggle_record()
            
        logger.info("pipeline_started")

    def _stop_pipeline(self):
        self.is_running = False
        
        if self._recorder and self._recorder.is_recording():
            self._recorder.stop()
            
        if self._processor:
            self._processor.stop()
            self._processor = None
            
        if self._source:
            self._source.release()
            self._source = None
            
        self.btn_toggle.config(text="▶ Start")
        self.btn_rec.config(state=tk.DISABLED, text="⏺ Record")
        
        # clear queue
        while not self.q_out.empty():
            try:
                self.q_out.get_nowait()
            except:
                break
                
        logger.info("pipeline_stopped")

    def _schedule_update(self):
        if self.is_running and self._annotator and self._source:
            try:
                # Try to get the latest processed result (with detections)
                result = None
                while not self.q_out.empty():
                    result = self.q_out.get_nowait()

                if result is not None and result.frame is not None and result.frame.size > 0:
                    # Annotate with bounding boxes
                    annotated = self._annotator.draw(result.frame, result)
                    self._fps_counter.tick()
                    annotated = self._annotator.draw_hud(
                        annotated,
                        fps=self._fps_counter.get_fps(),
                        source_label=str(self.source_var.get()),
                        recording=self._recorder.is_recording() if self._recorder else False
                    )
                    if self._recorder and self._recorder.is_recording():
                        self._recorder.write(annotated)
                    self._render_frame(annotated)

                else:
                    # q_out boş — ham frame'i direkt source'dan çek ve göster (siyah ekran olmasın)
                    try:
                        raw_ret, raw_frame = self._source._queue.get_nowait()
                        if raw_ret and raw_frame is not None and raw_frame.size > 0:
                            self._fps_counter.tick()
                            hud_frame = self._annotator.draw_hud(
                                raw_frame,
                                fps=self._fps_counter.get_fps(),
                                source_label=str(self.source_var.get()),
                                recording=self._recorder.is_recording() if self._recorder else False
                            )
                            self._render_frame(hud_frame)
                    except queue.Empty:
                        pass

            except Exception:
                pass

        # Loop every 15 ms
        self.root.after(15, self._schedule_update)

    def _render_frame(self, frame):
        """BGR numpy frame'i Tkinter canvas'a çizer."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self._current_photo = ImageTk.PhotoImage(image=img)
            self.canvas.config(width=img.width, height=img.height)
            self.canvas.create_image(0, 0, image=self._current_photo, anchor=tk.NW)
        except Exception:
            pass

    def on_close(self):
        self._stop_pipeline()
        self.root.destroy()
