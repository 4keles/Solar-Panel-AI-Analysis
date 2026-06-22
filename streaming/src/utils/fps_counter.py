import time
from collections import deque

class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        """Mark a new frame."""
        self._times.append(time.perf_counter())

    def get_fps(self) -> float:
        """Calculate FPS over the sliding window."""
        if len(self._times) < 2:
            return 0.0
        
        duration = self._times[-1] - self._times[0]
        if duration <= 0:
            return 0.0
            
        return len(self._times) / duration
