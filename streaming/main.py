#!/usr/bin/env python3
"""Solar Panel OD — Live Streaming Entry Point."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solar Panel Detection — Streaming")
    p.add_argument("--config", type=Path, default=Path("configs/streaming.yaml"))
    p.add_argument("--source", help="Override: 0, video.mp4, rtsp://...")
    p.add_argument("--model", type=Path, help="Override: path to .pt")
    p.add_argument("--record", action="store_true", help="Start recording immediately")
    p.add_argument("--no-gui", action="store_true", help="Headless mode (record only)")
    p.add_argument("--device", default=None, help="Override: cpu | 0 | cuda:0")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--dry-run", action="store_true", help="Load config only, don't start")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # Load main configure
    if not args.config.exists():
        logger.error("config_not_found", path=str(args.config))
        sys.exit(1)
        
    config = load_config(args.config)

    # Apply CLI Overrides
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
    if args.no_gui:
        config.setdefault("display", {})["show_window"] = False

    logger.info("config_loaded", config_path=str(args.config))

    if args.dry_run:
        print("Dry run OK. Config resolved to:")
        print(config)
        return

    if args.no_gui:
        logger.info("pipeline_startup", mode="headless")
        print("Headless mode is not fully implemented yet.")
    else:
        logger.info("pipeline_startup", mode="gui")
        import tkinter as tk
        from src.ui.control_panel import ControlPanel
        root = tk.Tk()
        app = ControlPanel(root, config)
        root.mainloop()

if __name__ == "__main__":
    main()
