import time
from pathlib import Path
from streaming.src.core.source_manager import ImageDirectorySource
from streaming.src.core.frame_processor import FrameProcessor
import queue

def main():
    src = ImageDirectorySource(Path("data/raw_data/captured"))
    src.open()
    
    q_out = queue.Queue()
    proc = FrameProcessor(
        model=None,
        device="cpu",
        conf=0.25,
        thermal_converter=None,
        q_in=src.get_queue(),
        q_out=q_out,
        target_fps=30
    )
    # mock model since we don't have it
    proc._model = type("MockModel", (), {"predict": lambda self, frame, conf, half: type("MockRes", (), {"boxes": type("MockBox", (), {"data": []})()})()})()
    proc.start()
    
    for _ in range(5):
        try:
            res = q_out.get(timeout=2.0)
            print("Got result! Frame shape:", res.frame.shape)
        except queue.Empty:
            print("Queue empty!")
    
    proc.stop()
    src.release()

if __name__ == "__main__":
    main()
