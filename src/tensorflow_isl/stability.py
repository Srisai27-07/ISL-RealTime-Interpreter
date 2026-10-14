"""Accept stable signs, reset on uncertainty, and avoid repeated held signs."""
from collections import deque


class StablePrediction:
    def __init__(self, frames=5):
        self.frames = frames
        self.history = deque(maxlen=frames)
        self.held = None

    def update(self, label):
        if label is None:
            self.history.clear()
            self.held = None
            return None, False
        self.history.append(label)
        if len(self.history) < self.frames or any(x != label for x in self.history):
            return None, False
        changed = self.held != label
        self.held = label
        return label, changed
