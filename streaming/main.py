#!/usr/bin/env python3
"""Solar Panel OD — Qt6 Live Streaming Entry Point."""

import argparse
import sys
from pathlib import Path

# Project root → scripts path
_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solar Panel OD — Qt6 Streaming")
    p.add_argument("--config", type=Path, default=Path("configs/streaming.yaml"))
    p.add_argument("--source", help="Override: 0 | video.mp4 | rtsp://...")
    p.add_argument("--model", type=Path, help="Override: path to .pt / .onnx / .engine")
    p.add_argument("--record", action="store_true", help="Start recording immediately")
    p.add_argument("--device", default=None, help="Override: cpu | 0 | cuda:0")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--dry-run", action="store_true", help="Validate config only, no GUI")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        logger.error("config_not_found", path=str(args.config))
        sys.exit(1)

    config = load_config(args.config)

    # CLI overrides
    if args.source is not None:
        if args.source.isdigit():
            config.setdefault("source", {})["camera_id"] = int(args.source)
            config["source"]["type"] = "camera"
        elif args.source.startswith("rtsp"):
            config.setdefault("source", {})["rtsp_url"] = args.source
            config["source"]["type"] = "rtsp"
        else:
            config.setdefault("source", {})["video_path"] = args.source
            config["source"]["type"] = "video"

    if args.model:
        config.setdefault("model", {})["path"] = str(args.model)
    if args.device:
        config.setdefault("model", {})["device"] = args.device
    if args.conf is not None:
        config.setdefault("model", {})["conf"] = args.conf
    if args.record:
        config.setdefault("recording", {})["enabled"] = True

    logger.info("config_loaded", path=str(args.config))

    if args.dry_run:
        print("Dry-run OK — Config:")
        import json
        print(json.dumps(config, indent=2, default=str))
        return

    # ── Launch Qt6 Application ──────────────────────────────────
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont, QFontDatabase
    from src.ui.control_panel import ControlPanel

    app = QApplication(sys.argv)
    app.setApplicationName("Solar Panel OD")
    app.setOrganizationName("SolarOD")

    # Load QSS theme
    qss_path = Path(__file__).parent / "src" / "ui" / "theme.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    window = ControlPanel(config, config_path=args.config)
    window.show()

    logger.info("qt6_app_started")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
