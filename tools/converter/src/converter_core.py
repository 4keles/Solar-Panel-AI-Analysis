"""converter_core.py — Model conversion engine.

Converts .pt YOLO models to faster inference formats:
  Priority: TensorRT .engine → ONNX → TorchScript

Each converter is tried in order and the first successful one is used,
unless the user specifies a target format explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

# ─────────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────────

class OutputFormat(str, Enum):
    TENSORRT = "engine"
    ONNX     = "onnx"
    TORCHSCRIPT = "torchscript"

    @classmethod
    def from_str(cls, s: str) -> "OutputFormat":
        s = s.lower().strip(".")
        mapping = {"engine": cls.TENSORRT, "trt": cls.TENSORRT,
                   "onnx": cls.ONNX, "torchscript": cls.TORCHSCRIPT, "ts": cls.TORCHSCRIPT}
        if s not in mapping:
            raise ValueError(f"Bilinmeyen format: '{s}'. Seçenekler: {list(mapping.keys())}")
        return mapping[s]


@dataclass
class ConversionConfig:
    input_path:   Path
    output_dir:   Path
    format:       OutputFormat
    imgsz:        int   = 640
    half:         bool  = True    # FP16 — GPU'da büyük hız artışı
    batch:        int   = 1
    device:       str   = "0"     # "0" = cuda:0, "cpu" = CPU
    workspace_gb: int   = 4       # TRT workspace (GB)
    simplify:     bool  = True    # ONNX graph simplification
    opset:        int   = 17      # ONNX opset version


@dataclass
class ConversionResult:
    success:      bool
    output_path:  Path | None
    format:       OutputFormat
    elapsed_sec:  float
    message:      str
    warnings:     list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
#  Core Converter
# ─────────────────────────────────────────────────────────────

class ModelConverter:
    """Converts Ultralytics .pt models to optimized formats.
    
    Usage:
        converter = ModelConverter(log_callback=print)
        result = converter.convert(config)
    """

    def __init__(self, log_callback: Callable[[str], None] | None = None):
        """
        Args:
            log_callback: Optional function to receive progress messages.
                          Receives a single string per message.
        """
        self._log = log_callback or print

    # ── Public API ────────────────────────────────────────────

    def convert(self, cfg: ConversionConfig) -> ConversionResult:
        """Run conversion according to config. Returns ConversionResult."""
        self._log(f"[INFO] Dönüştürme başlıyor: {cfg.input_path.name} → {cfg.format.value}")
        self._log(f"[INFO] Parametreler: imgsz={cfg.imgsz}, half={cfg.half}, "
                  f"batch={cfg.batch}, device={cfg.device}")

        if not cfg.input_path.exists():
            return ConversionResult(
                success=False, output_path=None, format=cfg.format,
                elapsed_sec=0, message=f"Dosya bulunamadı: {cfg.input_path}"
            )

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        try:
            if cfg.format == OutputFormat.TENSORRT:
                result = self._to_tensorrt(cfg)
            elif cfg.format == OutputFormat.ONNX:
                result = self._to_onnx(cfg)
            elif cfg.format == OutputFormat.TORCHSCRIPT:
                result = self._to_torchscript(cfg)
            else:
                raise ValueError(f"Desteklenmeyen format: {cfg.format}")

            result.elapsed_sec = time.perf_counter() - t0
            return result

        except Exception as e:
            elapsed = time.perf_counter() - t0
            self._log(f"[ERROR] Dönüştürme başarısız: {e}")
            return ConversionResult(
                success=False, output_path=None, format=cfg.format,
                elapsed_sec=elapsed, message=str(e)
            )

    def detect_system(self) -> dict:
        """Return system capabilities dict for display in GUI."""
        info: dict = {}
        try:
            import torch
            info["torch"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["cuda_version"] = torch.version.cuda
                mem_total = torch.cuda.get_device_properties(0).total_memory
                info["vram_gb"] = round(mem_total / 1024**3, 1)
        except Exception as e:
            info["torch_error"] = str(e)

        try:
            import tensorrt as trt  # type: ignore
            info["tensorrt"] = trt.__version__
        except ImportError:
            info["tensorrt"] = None

        try:
            import onnx
            info["onnx"] = onnx.__version__
        except ImportError:
            info["onnx"] = None

        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            info["onnxruntime"] = ort.__version__
            info["onnxruntime_gpu"] = "CUDAExecutionProvider" in providers
        except ImportError:
            info["onnxruntime"] = None

        return info

    # ── Private Converters ────────────────────────────────────

    def _to_tensorrt(self, cfg: ConversionConfig) -> ConversionResult:
        """Export to TensorRT .engine via Ultralytics."""
        self._log("[INFO] TensorRT engine dönüşümü başlatılıyor...")
        self._log("[INFO] Bu işlem GPU ve model boyutuna göre 2-15 dakika sürebilir.")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(f"Ultralytics kurulu değil: {e}") from e

        # Validate TRT available
        try:
            import tensorrt  # type: ignore  # noqa: F401
        except ImportError:
            self._log("[WARNING] TensorRT Python paketi bulunamadı.")
            self._log("[WARNING] Ultralytics kendi TRT builder'ını kullanmaya çalışacak.")

        model = YOLO(str(cfg.input_path))
        output_path = cfg.output_dir / (cfg.input_path.stem + ".engine")

        self._log(f"[INFO] Hedef: {output_path}")

        exported = model.export(
            format="engine",
            imgsz=cfg.imgsz,
            half=cfg.half,
            batch=cfg.batch,
            device=cfg.device,
            workspace=cfg.workspace_gb,
            verbose=False,
        )

        # Ultralytics places the file next to the .pt; move it to output_dir
        exported_path = Path(str(exported))
        if exported_path.exists() and exported_path != output_path:
            import shutil
            shutil.move(str(exported_path), str(output_path))

        self._log(f"[OK] Engine oluşturuldu: {output_path}")
        return ConversionResult(
            success=True, output_path=output_path, format=cfg.format,
            elapsed_sec=0, message="TensorRT engine başarıyla oluşturuldu."
        )

    def _to_onnx(self, cfg: ConversionConfig) -> ConversionResult:
        """Export to ONNX via Ultralytics."""
        self._log("[INFO] ONNX dönüşümü başlatılıyor...")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(f"Ultralytics kurulu değil: {e}") from e

        model = YOLO(str(cfg.input_path))
        output_path = cfg.output_dir / (cfg.input_path.stem + ".onnx")

        self._log(f"[INFO] Hedef: {output_path}")

        exported = model.export(
            format="onnx",
            imgsz=cfg.imgsz,
            half=cfg.half and cfg.device != "cpu",
            batch=cfg.batch,
            simplify=cfg.simplify,
            opset=cfg.opset,
            verbose=False,
        )

        exported_path = Path(str(exported))
        if exported_path.exists() and exported_path != output_path:
            import shutil
            shutil.move(str(exported_path), str(output_path))

        # Optional ONNX simplification
        if cfg.simplify:
            self._simplify_onnx(output_path)

        size_mb = output_path.stat().st_size / 1024**2
        self._log(f"[OK] ONNX oluşturuldu: {output_path}  ({size_mb:.1f} MB)")
        return ConversionResult(
            success=True, output_path=output_path, format=cfg.format,
            elapsed_sec=0, message=f"ONNX başarıyla oluşturuldu. Boyut: {size_mb:.1f} MB"
        )

    def _to_torchscript(self, cfg: ConversionConfig) -> ConversionResult:
        """Export to TorchScript via Ultralytics."""
        self._log("[INFO] TorchScript dönüşümü başlatılıyor...")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(f"Ultralytics kurulu değil: {e}") from e

        model = YOLO(str(cfg.input_path))
        output_path = cfg.output_dir / (cfg.input_path.stem + ".torchscript")

        self._log(f"[INFO] Hedef: {output_path}")

        exported = model.export(
            format="torchscript",
            imgsz=cfg.imgsz,
            verbose=False,
        )

        exported_path = Path(str(exported))
        if exported_path.exists() and exported_path != output_path:
            import shutil
            shutil.move(str(exported_path), str(output_path))

        size_mb = output_path.stat().st_size / 1024**2
        self._log(f"[OK] TorchScript oluşturuldu: {output_path}  ({size_mb:.1f} MB)")
        return ConversionResult(
            success=True, output_path=output_path, format=cfg.format,
            elapsed_sec=0, message=f"TorchScript başarıyla oluşturuldu. Boyut: {size_mb:.1f} MB"
        )

    def _simplify_onnx(self, path: Path) -> None:
        """Attempt onnxsim simplification; skip if not installed."""
        try:
            import onnx
            import onnxsim  # type: ignore
            self._log("[INFO] ONNX grafik basitleştirme (onnxsim) uygulanıyor...")
            model_onnx = onnx.load(str(path))
            simplified, ok = onnxsim.simplify(model_onnx)
            if ok:
                onnx.save(simplified, str(path))
                self._log("[OK] ONNX basitleştirme tamamlandı.")
            else:
                self._log("[WARNING] onnxsim başarısız oldu, orijinal ONNX korunuyor.")
        except ImportError:
            self._log("[WARNING] onnxsim kurulu değil — basitleştirme atlandı.")
        except Exception as e:
            self._log(f"[WARNING] ONNX basitleştirme hatası: {e}")
