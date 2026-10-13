"""
Real-Time Prediction Pipeline with Sliding Window, Debouncing, and Sentence Building.
Ensures stable, jitter-free sign recognition during live video streaming.
"""

from typing import Optional, List, Tuple, Callable
from collections import deque
import time
import numpy as np
import torch

from src.models.temporal_model import ISLBiLSTM, get_optimal_device


class RealTimePredictor:
    """
    Manages temporal landmark buffering, inference smoothing, debouncing,
    and cumulative sentence assembly for live sign language translation.
    """

    def __init__(
        self,
        model: ISLBiLSTM,
        class_names: List[str],
        sequence_length: int = 30,
        prediction_threshold: float = 0.80,
        debounce_frames: int = 10,
        cooldown_seconds: float = 1.5,
        on_word_committed: Optional[Callable[[str], None]] = None
    ):
        self.model = model
        self.class_names = class_names
        self.sequence_length = sequence_length
        self.prediction_threshold = prediction_threshold
        self.debounce_frames = debounce_frames
        self.cooldown_seconds = cooldown_seconds
        self.on_word_committed = on_word_committed

        self.device = get_optimal_device()
        self.model.to(self.device)
        self.model.eval()

        # Landmark history buffer: shape (sequence_length, feature_dim)
        self.buffer = deque(maxlen=sequence_length)
        # Recent prediction class indices for debouncing
        self.recent_predictions = deque(maxlen=debounce_frames)

        # State tracking
        self.current_prediction: Optional[str] = None
        self.current_confidence: float = 0.0
        self.last_committed_word: Optional[str] = None
        self.last_commit_time: float = 0.0
        self.sentence: List[str] = []

    def update(self, landmark_vector: np.ndarray) -> Tuple[Optional[str], float, List[str]]:
        """
        Ingests a new frame's landmark vector (126,).
        
        Returns:
            (current_prediction, current_confidence, sentence_words)
        """
        # Append landmark to temporal buffer
        self.buffer.append(landmark_vector)

        # We can only perform temporal inference once the buffer is full
        if len(self.buffer) < self.sequence_length:
            return None, 0.0, self.sentence

        # Check if hands are actually present (if vector is all zeros, user is resting)
        if np.all(landmark_vector == 0):
            self.current_prediction = None
            self.current_confidence = 0.0
            return None, 0.0, self.sentence

        # Prepare tensor: (1, seq_len, input_size)
        seq_array = np.array(self.buffer, dtype=np.float32)
        tensor_in = torch.from_numpy(seq_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            proba = self.model.predict_proba(tensor_in).cpu().numpy()[0]

        max_idx = int(np.argmax(proba))
        confidence = float(proba[max_idx])
        predicted_class = self.class_names[max_idx]

        self.current_prediction = predicted_class
        self.current_confidence = confidence

        # Apply debouncing & thresholding
        if confidence >= self.prediction_threshold:
            self.recent_predictions.append(max_idx)
        else:
            self.recent_predictions.append(-1)

        # Check if the buffer has consensus
        if len(self.recent_predictions) == self.debounce_frames:
            # Check if all or majority match the current max_idx
            counts = [p for p in self.recent_predictions if p == max_idx]
            if len(counts) >= (self.debounce_frames * 0.75):
                now = time.time()
                # Ensure cooldown or different sign to avoid runaway repetition
                is_different = (predicted_class != self.last_committed_word)
                time_elapsed = (now - self.last_commit_time) >= self.cooldown_seconds

                if is_different or time_elapsed:
                    self._commit_word(predicted_class)
                    self.last_commit_time = now

        return self.current_prediction, self.current_confidence, self.sentence

    def _commit_word(self, word: str):
        self.last_committed_word = word
        self.sentence.append(word)
        if self.on_word_committed:
            try:
                self.on_word_committed(word)
            except Exception:
                pass

    def clear_sentence(self):
        """Clears the constructed sentence."""
        self.sentence.clear()
        self.last_committed_word = None

    def get_sentence_text(self) -> str:
        """Returns the accumulated sentence as a string."""
        return " ".join(self.sentence)
