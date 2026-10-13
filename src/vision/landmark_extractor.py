"""
Landmark Extractor Module for Real-Time Indian Sign Language (ISL).
Extracts 21 3D landmarks per hand (Left + Right) using MediaPipe Holistic/Hands.
ISL inherently relies on two-handed gestures and relative hand positioning.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp


class ISLLandmarkExtractor:
    """
    Extracts and normalizes dual-hand landmarks for ISL recognition.
    Feature vector size:
      - Left Hand:  21 landmarks * (x, y, z) = 63
      - Right Hand: 21 landmarks * (x, y, z) = 63
      - Total Hand Features: 126 floats
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.4,
        min_tracking_confidence: float = 0.4,
        include_pose: bool = False
    ):
        self.include_pose = include_pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, any]:
        """
        Processes a BGR video frame and extracts dual-hand landmark feature vector.
        Matches frontend exactly:
          Indices 0..62:  Left Hand (or first hand) (21 landmarks * 3 coords)
          Indices 63..125: Right Hand (or second hand) (21 landmarks * 3 coords)
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        vec = np.zeros(126, dtype=np.float32)

        if results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                # Determine hand slot (left vs right)
                is_right = False
                if results.multi_handedness and idx < len(results.multi_handedness):
                    label = results.multi_handedness[idx].classification[0].label
                    is_right = (label == "Right")
                else:
                    is_right = (idx == 1)

                offset = 63 if is_right else 0

                coords = []
                for lm in hand_lms.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                
                vec[offset : offset + 63] = np.array(coords[:63], dtype=np.float32)

        return vec, results

    def draw_landmarks(self, frame_bgr: np.ndarray, results: any) -> np.ndarray:
        """
        Draws skeleton overlays on the frame for visual feedback.
        """
        output = frame_bgr.copy()

        if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    output,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        return output

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
