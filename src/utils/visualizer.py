"""
Visual HUD and Interface Overlays for Real-Time ISL Interpreter.
Renders confidence bars, active sign labels, sentence accumulation, and FPS.
"""

from typing import List, Optional
import cv2
import numpy as np


class HUDVisualizer:
    """
    Renders professional broadcast-style HUD overlays onto camera frames.
    """

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        prediction: Optional[str],
        confidence: float,
        sentence: List[str],
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Draws top status bar, bottom sentence tray, and confidence meter.
        """
        h, w = frame.shape[:2]
        output = frame.copy()

        # 1. Top HUD Overlay: Active Gesture & Confidence
        top_overlay = output.copy()
        cv2.rectangle(top_overlay, (0, 0), (w, 65), (20, 20, 20), -1)
        output = cv2.addWeighted(top_overlay, 0.75, output, 0.25, 0)

        # Draw Prediction Label
        label_text = f"Sign: {prediction if prediction else 'Detecting...'}"
        cv2.putText(output, label_text, (20, 42), self.font, 0.9, (0, 255, 200), 2, cv2.LINE_AA)

        # Draw FPS in top right
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(output, fps_text, (w - 140, 42), self.font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        # Draw Confidence Progress Bar
        bar_x, bar_y, bar_w, bar_h = 240, 20, 160, 20
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
        fill_w = int(bar_w * np.clip(confidence, 0.0, 1.0))
        bar_color = (0, 220, 100) if confidence >= 0.8 else (0, 165, 255)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.putText(output, f"{int(confidence * 100)}%", (bar_x + bar_w + 10, bar_y + 16), self.font, 0.55, (255, 255, 255), 1)

        # 2. Bottom Sentence Accumulator Tray
        bottom_overlay = output.copy()
        cv2.rectangle(bottom_overlay, (0, h - 60), (w, h), (15, 15, 15), -1)
        output = cv2.addWeighted(bottom_overlay, 0.85, output, 0.15, 0)

        sentence_str = " ".join(sentence) if sentence else "[Perform signs to construct sentence...]"
        cv2.putText(output, "Sentence:", (15, h - 22), self.font, 0.65, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(output, sentence_str, (120, h - 22), self.font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        return output
