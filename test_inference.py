import cv2
import queue
import time
from pathlib import Path
from ultralytics import YOLO
from streaming.src.core.frame_processor import FrameProcessor

def main():
    model = YOLO("models/v1.0.4/best.engine", task="detect")
    q_in = queue.Queue()
    q_out = queue.Queue()
    
    proc = FrameProcessor(
        model=model,
        input_queue=q_in,
        output_queue=q_out,
        device="cuda:0",
    )
    proc.start()
    
    img = cv2.imread("data/raw_data/captured/test_capture_1.jpg", cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Could not read image")
        return
        
    print(f"Image read. Shape: {img.shape}")
    q_in.put((True, img))
    
    try:
        res = q_out.get(timeout=3.0)
        print("Got result! Frame shape:", res.frame.shape)
    except queue.Empty:
        print("Queue empty! Process failed!")
        
    proc.stop()

if __name__ == "__main__":
    main()
