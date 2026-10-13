"""
Diagnostic Script: Test Webcam & MediaPipe Dual-Hand Landmark Tracking.
Run this first to verify camera hardware and library installations.
Usage:
    python scripts/test_camera.py
"""

import sys
import time
import cv2

# Add root directory to sys.path
sys.path.append(".")

from src.capture.camera_stream import CameraStream
from src.vision.landmark_extractor import ISLLandmarkExtractor


def main():
    print("=" * 60)
    print("Starting Webcam & MediaPipe Diagnostic Test...")
    print("Press 'q' or 'ESC' in the video window to exit.")
    print("=" * 60)

    try:
        stream = CameraStream(src=0, width=640, height=480).start()
    except Exception as e:
        print(f"[ERROR] Failed to access webcam: {e}")
        print("Please check camera permissions in System Settings -> Privacy & Security -> Camera.")
        return

    extractor = ISLLandmarkExtractor(min_detection_confidence=0.6, min_tracking_confidence=0.5)

    prev_time = time.time()
    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Mirror view for natural interaction
            frame = cv2.flip(frame, 1)

            # Process landmarks
            features, results = extractor.process_frame(frame)
            annotated_frame = extractor.draw_landmarks(frame, results)

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # Status overlays
            lh_detected = results.left_hand_landmarks is not None
            rh_detected = results.right_hand_landmarks is not None

            status_text = f"FPS: {fps:.1f} | Left Hand: {'YES' if lh_detected else 'NO'} | Right Hand: {'YES' if rh_detected else 'NO'}"
            cv2.putText(annotated_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Feature Vector Size: {features.shape[0]}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            cv2.imshow("ISL Diagnostic Test - Press Q to Quit", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        stream.stop()
        extractor.close()
        cv2.destroyAllWindows()
        print("Diagnostic test closed cleanly.")


if __name__ == "__main__":
    main()
