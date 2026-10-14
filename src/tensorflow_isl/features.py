"""Shared offline/webcam features preserving relative positions of both hands."""
from pathlib import Path
import numpy as np

FEATURE_VERSION = "spatial-hands-v1"
FEATURE_DIM = 128


def normalize_hands(hands):
    """Hands are (21,3) coordinates in the same square image, not independently scaled.

    Spatial ordering avoids dependence on handedness labels or detection order.
    MediaPipe z is wrist-relative per hand, so it is NOT inter-hand depth.
    """
    if not hands:
        return None
    if len(hands) > 2:
        raise ValueError("At most two hands are supported")
    arrays = [np.asarray(h, dtype=np.float32).reshape(21, 3) for h in hands]
    if any(not np.isfinite(h).all() for h in arrays):
        raise ValueError("Non-finite landmarks")
    arrays.sort(key=lambda h: (float(h[0, 0]), float(h[0, 1])))
    lengths = [np.linalg.norm(h[9, :2] - h[0, :2]) for h in arrays]
    scale = float(np.mean(lengths))
    if scale < 1e-5:
        return None
    center = np.mean([h[0, :2] for h in arrays], axis=0)
    result = np.zeros(FEATURE_DIM, dtype=np.float32)
    for i, h in enumerate(arrays):
        h = h.copy()
        h[:, :2] = (h[:, :2] - center) / scale
        h[:, 2] = (h[:, 2] - h[0, 2]) / scale
        result[i * 63:(i + 1) * 63] = h.ravel()
        result[126 + i] = 1
    return result


def square_frame(frame):
    """Pad to a square instead of distorting aspect ratio; shared by every input."""
    import cv2
    height, width = frame.shape[:2]
    side = max(height, width)
    return cv2.copyMakeBorder(frame, (side-height)//2, (side-height+1)//2,
                             (side-width)//2, (side-width+1)//2,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))


def browser_features(hands, width, height):
    """Map unmirrored browser landmarks to square-padded image coordinates."""
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Invalid frame dimensions")
    side = max(width, height)
    arrays = []
    for values in hands:
        h = np.asarray(values, dtype=np.float32)
        if h.shape != (21, 3):
            raise ValueError("Each hand must have 21 xyz landmarks")
        h = h.copy()
        h[:, 0] = h[:, 0]*width/side + ((side-width)//2)/side
        h[:, 1] = h[:, 1]*height/side + ((side-height)//2)/side
        h[:, 2] *= width/side
        arrays.append(h)
    return normalize_hands(arrays)


class HandExtractor:
    def __init__(self, model_path, video=False, detection_confidence=0.4):
        import mediapipe as mp
        self.mp = mp
        self.video = video
        self.last_timestamp = -1
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Missing Hand Landmarker model: {model_path}")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=(mp.tasks.vision.RunningMode.VIDEO if video
                          else mp.tasks.vision.RunningMode.IMAGE),
            num_hands=2, min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=0.4, min_tracking_confidence=0.5)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def extract(self, frame, timestamp_ms=None):
        import cv2
        square = square_frame(frame)
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb))
        if self.video:
            timestamp_ms = max(int(timestamp_ms), self.last_timestamp + 1)
            self.last_timestamp = timestamp_ms
            result = self.detector.detect_for_video(image, timestamp_ms)
        else:
            result = self.detector.detect(image)
        hands = [np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32)
                 for hand in result.hand_landmarks]
        return normalize_hands(hands), hands, square

    def close(self):
        self.detector.close()
