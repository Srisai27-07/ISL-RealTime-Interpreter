"""
Dataset Collection Tool for ISL Gestures.
Designed for Teammate #1 to collect custom ISL landmark sequences directly from webcam.

Workflow:
1. Prompts for target class (e.g. 'Hello', 'Thank You').
2. Counts down on screen (3.. 2.. 1..).
3. Records 30 consecutive frames of 3D landmarks (126 features/frame).
4. Saves directly as `data/processed/<class>/seq_<timestamp>.npy`.

Usage:
    python scripts/01_collect_data.py --samples 25
"""

import os
import sys
import time
import argparse
import yaml
import cv2
import numpy as np

# Add project root to sys.path
sys.path.append(".")

from src.capture.camera_stream import CameraStream
from src.vision.landmark_extractor import ISLLandmarkExtractor


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def record_dataset(samples_per_class: int = 25, target_class: str = None):
    config = load_config()
    classes = [target_class] if target_class else config["classes"]
    seq_length = config["sequence"]["sequence_length"]
    save_root = "data/processed"

    print("=" * 65)
    print(" Indian Sign Language (ISL) Dataset Collection Studio")
    print(f" Target Classes: {classes}")
    print(f" Samples per Class: {samples_per_class}")
    print(f" Frames per Sequence: {seq_length}")
    print(" Press SPACE to begin sequence, 's' to skip, 'q' to quit.")
    print("=" * 65)

    stream = CameraStream(src=0, width=640, height=480).start()
    extractor = ISLLandmarkExtractor()

    try:
        for class_name in classes:
            class_dir = os.path.join(save_root, class_name)
            os.makedirs(class_dir, exist_ok=True)
            existing_count = len([f for f in os.listdir(class_dir) if f.endswith(".npy")])

            for sample_idx in range(existing_count, existing_count + samples_per_class):
                # 1. Waiting for user readiness
                ready = False
                while not ready:
                    ret, frame = stream.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    features, results = extractor.process_frame(frame)
                    annotated = extractor.draw_landmarks(frame, results)

                    # Prompt banner
                    cv2.rectangle(annotated, (0, 0), (640, 70), (0, 0, 0), -1)
                    prompt = f"Sign: [{class_name}] | Sample {sample_idx + 1}/{existing_count + samples_per_class}"
                    cv2.putText(annotated, prompt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(annotated, "Press [SPACE] when ready to record | [q] to Quit", (20, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                    cv2.imshow("ISL Data Recorder", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:  # Space
                        ready = True
                    elif key == ord('q'):
                        return
                    elif key == ord('s'):
                        break

                # 2. Countdown: 3, 2, 1
                for count in [3, 2, 1]:
                    start_time = time.time()
                    while time.time() - start_time < 0.7:
                        ret, frame = stream.read()
                        if not ret:
                            continue
                        frame = cv2.flip(frame, 1)
                        _, results = extractor.process_frame(frame)
                        annotated = extractor.draw_landmarks(frame, results)

                        cv2.putText(annotated, f"Starting in: {count}", (200, 260),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 140, 255), 4)
                        cv2.imshow("ISL Data Recorder", annotated)
                        cv2.waitKey(1)

                # 3. Recording sequence of 30 frames
                sequence_landmarks = []
                frame_count = 0
                while frame_count < seq_length:
                    ret, frame = stream.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    features, results = extractor.process_frame(frame)
                    sequence_landmarks.append(features)
                    annotated = extractor.draw_landmarks(frame, results)

                    # Recording HUD
                    cv2.circle(annotated, (30, 40), 12, (0, 0, 255), -1)
                    cv2.putText(annotated, f"RECORDING [{class_name}]: {frame_count + 1}/{seq_length}",
                                (55, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("ISL Data Recorder", annotated)
                    cv2.waitKey(1)
                    frame_count += 1

                # 4. Save numpy array: shape (30, 126)
                seq_np = np.array(sequence_landmarks, dtype=np.float32)
                timestamp = int(time.time() * 1000)
                out_path = os.path.join(class_dir, f"seq_{timestamp}.npy")
                np.save(out_path, seq_np)
                print(f"[SAVED] {class_name} sample {sample_idx + 1} -> {out_path}")
                time.sleep(0.3)

    finally:
        stream.stop()
        extractor.close()
        cv2.destroyAllWindows()
        print("\nData collection finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record ISL gesture dataset sequences.")
    parser.add_argument("--samples", type=int, default=20, help="Number of sequences per class")
    parser.add_argument("--class-name", type=str, default=None, help="Target specific class")
    args = parser.parse_args()

    record_dataset(samples_per_class=args.samples, target_class=args.class_name)
