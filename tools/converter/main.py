#!/usr/bin/env python3
"""Solar Panel OD — Model Converter Tool.

GUI arayüzü:
    python tools/converter/main.py

CLI (headless) modu:
    python tools/converter/main.py --cli \\
        --input models/v1.0.4/best.pt \\
        --format onnx \\
        --imgsz 640 \\
        --half
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
_project_root = Path(__file__).resolve().parent.parent.parent
_scripts_dir = _project_root / "scripts"
_converter_src = Path(__file__).resolve().parent / "src"

if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
if str(_converter_src) not in sys.path:
    sys.path.insert(0, str(_converter_src))

_models_dir = _project_root / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Solar Panel OD — Model Dönüştürücü",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # GUI başlat:
  python tools/converter/main.py

  # CLI: .pt → .onnx
  python tools/converter/main.py --cli --input models/v1.0.4/best.pt --format onnx

  # CLI: .pt → TensorRT engine (GPU)
  python tools/converter/main.py --cli --input models/v1.0.4/best.pt --format engine --half

  # CLI: tam parametreler
  python tools/converter/main.py --cli \\
      --input models/v1.0.4/best.pt \\
      --format onnx \\
      --imgsz 640 \\
      --output models/v1.0.4/
        """,
    )

    p.add_argument("--cli", action="store_true",
                   help="GUI olmadan CLI modunda çalıştır")
    p.add_argument("--input", type=Path,
                   help="Giriş .pt model yolu")
    p.add_argument("--format", choices=["engine", "onnx", "torchscript"],
                   default="onnx", help="Hedef format (varsayılan: onnx)")
    p.add_argument("--output", type=Path,
                   help="Çıktı dizini (varsayılan: input modelin dizini)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Görüntü boyutu (varsayılan: 640)")
    p.add_argument("--half", action="store_true",
                   help="FP16 kullan (GPU'da daha hızlı)")
    p.add_argument("--batch", type=int, default=1,
                   help="Batch size (varsayılan: 1)")
    p.add_argument("--device", default="0",
                   help="CUDA device (varsayılan: 0 = cuda:0, cpu = CPU)")
    p.add_argument("--workspace", type=int, default=4,
                   help="TensorRT workspace GB (varsayılan: 4)")
    p.add_argument("--no-simplify", action="store_true",
                   help="ONNX simplify'ı devre dışı bırak")
    return p.parse_args()


def run_cli(args: argparse.Namespace) -> int:
    """CLI (headless) conversion mode. Returns exit code."""
    from converter_core import ModelConverter, ConversionConfig, OutputFormat

    if not args.input:
        print("[ERROR] --input parametresi gerekli. Kullanım: --help")
        return 1

    input_path = args.input
    if not input_path.is_absolute():
        input_path = _project_root / input_path

    if not input_path.exists():
        print(f"[ERROR] Model bulunamadı: {input_path}")
        return 1

    output_dir = args.output or input_path.parent
    if not output_dir.is_absolute():
        output_dir = _project_root / output_dir

    fmt = OutputFormat.from_str(args.format)

    cfg = ConversionConfig(
        input_path=input_path,
        output_dir=output_dir,
        format=fmt,
        imgsz=args.imgsz,
        half=args.half,
        batch=args.batch,
        device=args.device,
        workspace_gb=args.workspace,
        simplify=not args.no_simplify,
    )

    converter = ModelConverter(log_callback=print)
    result = converter.convert(cfg)

    if result.success:
        print(f"\n✅  Tamamlandı: {result.output_path}  ({result.elapsed_sec:.1f}s)")
        return 0
    else:
        print(f"\n❌  Başarısız: {result.message}")
        return 1


def run_gui() -> int:
    """Launch Qt6 GUI."""
    from PyQt6.QtWidgets import QApplication
    from ui.converter_panel import ConverterPanel

    app = QApplication(sys.argv)
    app.setApplicationName("Solar Panel OD — Converter")

    # Load shared QSS theme if available
    streaming_qss = _project_root / "streaming" / "src" / "ui" / "theme.qss"
    if streaming_qss.exists():
        app.setStyleSheet(streaming_qss.read_text(encoding="utf-8"))

    window = ConverterPanel(models_dir=_models_dir)
    window.show()
    return app.exec()


def main() -> None:
    args = parse_args()
    if args.cli:
        sys.exit(run_cli(args))
    else:
        sys.exit(run_gui())


if __name__ == "__main__":
    main()
