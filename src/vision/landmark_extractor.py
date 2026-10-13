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
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        include_pose: bool = False
    ):
        self.include_pose = include_pose
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1
        )

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, any]:
        """
        Processes a BGR video frame and extracts the landmark feature vector.
        
        Args:
            frame_bgr: OpenCV image (H, W, 3) in BGR color space.
            
        Returns:
            features: 1D numpy array of shape (126,) [or (198,) if pose included]
            results: MediaPipe detection results object for rendering.
        """
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Extract Left Hand (21 landmarks * 3 coordinates)
        if results.left_hand_landmarks:
            lh = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            ).flatten()
        else:
            lh = np.zeros(21 * 3, dtype=np.float32)

        # Extract Right Hand (21 landmarks * 3 coordinates)
        if results.right_hand_landmarks:
            rh = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            ).flatten()
        else:
            rh = np.zeros(21 * 3, dtype=np.float32)

        if self.include_pose:
            # Extract upper body pose (33 landmarks * 4: x, y, z, visibility) or first 24
            if results.pose_landmarks:
                pose = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                ).flatten()
            else:
                pose = np.zeros(33 * 3, dtype=np.float32)
            features = np.concatenate([lh, rh, pose]).astype(np.float32)
        else:
            features = np.concatenate([lh, rh]).astype(np.float32)

        return features, results

    def draw_landmarks(self, frame_bgr: np.ndarray, results: any) -> np.ndarray:
        """
        Draws skeleton overlays on the frame for visual feedback.
        """
        output = frame_bgr.copy()

        # Draw left hand connections
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                output,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        # Draw right hand connections
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                output,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        # Draw pose connections (upper body)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                output,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        return output

    def close(self):
        """Release MediaPipe resources."""
        self.holistic.close()
