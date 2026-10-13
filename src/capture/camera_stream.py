"""
Threaded Webcam Stream for Low-Latency Real-Time Video Ingestion.
Prevents frame buffer lag when coupled with heavy ML inference.
"""

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class CameraStream:
    """
    Spawns a background thread that constantly reads the latest webcam frame.
    Guarantees that inference always runs on the freshest frame with zero latency buffer.
    """

    def __init__(self, src: int = 0, width: int = 640, height: int = 480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self) -> "CameraStream":
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video capture device at index {self.src}")
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
