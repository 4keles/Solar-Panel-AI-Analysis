import sys
import time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from streaming.src.ui.control_panel import ControlPanel
from streaming.src.core.source_manager import SourceType
from pathlib import Path
from scripts.utils.config_loader import load_config

def main():
    app = QApplication(sys.argv)
    config = load_config(Path("configs/streaming.yaml"))
    panel = ControlPanel(config)
    
    def run_test():
        print("Running test...")
        panel.source_selector.combo_type.setCurrentText("Resim Klasörü (.jpg, .png)")
        panel.source_selector.input_path.setText(str(Path("data/raw_data/captured").absolute()))
        
        # Click start
        panel.btn_start.click()
        
        def check_status():
            print(f"Is running: {panel.is_running}")
            print(f"Has Pixmap: {panel.video_widget._current_pixmap is not None}")
            print(f"Queue size: {panel._source.get_queue().qsize()}")
            app.quit()
            
        QTimer.singleShot(2000, check_status)
        
    QTimer.singleShot(500, run_test)
    app.exec()

if __name__ == "__main__":
    main()
