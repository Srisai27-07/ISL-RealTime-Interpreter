"""
Standalone Real-Time Desktop ISL Interpreter.
High-performance OpenCV + MediaPipe + PyTorch inference pipeline with HUD and TTS audio.

Usage:
    python app/live_demo.py
"""

import sys
import os
import time
import yaml
import cv2

sys.path.append(".")

from src.capture.camera_stream import CameraStream
from src.vision.landmark_extractor import ISLLandmarkExtractor
from src.models.temporal_model import load_checkpoint
from src.pipeline.real_time_predictor import RealTimePredictor
from src.utils.visualizer import HUDVisualizer
from src.utils.text_to_speech import TextToSpeechEngine


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    weights_path = config["model"]["weights_path"]
    if not os.path.exists(weights_path):
        print(f"[ERROR] Model weights not found at '{weights_path}'.")
        print("Please train a model first using:")
        print("    python scripts/03_train_model.py --generate-mock")
        return

    print("=" * 60)
    print("Loading ISL Real-Time Interpreter...")
    print(f"Loading model checkpoint: {weights_path}")
    model, class_names = load_checkpoint(weights_path)
    print(f"Loaded {len(class_names)} classes: {class_names}")

    tts = TextToSpeechEngine(enabled=config["tts"]["enabled"], rate=config["tts"]["rate"])
    visualizer = HUDVisualizer()

    predictor = RealTimePredictor(
        model=model,
        class_names=class_names,
        sequence_length=config["sequence"]["sequence_length"],
        prediction_threshold=config["sequence"]["prediction_threshold"],
        debounce_frames=config["sequence"]["debounce_frames"],
        cooldown_seconds=config["sequence"]["cooldown_seconds"],
        on_word_committed=lambda word: tts.speak(word)
    )

    extractor = ISLLandmarkExtractor(
        min_detection_confidence=config["vision"]["min_detection_confidence"],
        min_tracking_confidence=config["vision"]["min_tracking_confidence"]
    )

    stream = CameraStream(
        src=config["camera"]["device_index"],
        width=config["camera"]["width"],
        height=config["camera"]["height"]
    ).start()

    print("\n[READY] Video feed started. Controls:")
    print("  [c] : Clear sentence")
    print("  [s] : Speak current sentence aloud")
    print("  [q] : Quit application")
    print("=" * 60)

    prev_time = time.time()
    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)

            # 1. Landmark extraction
            features, results = extractor.process_frame(frame)
            annotated = extractor.draw_landmarks(frame, results)

            # 2. Pipeline inference & debouncing
            pred_class, confidence, sentence = predictor.update(features)

            # 3. FPS calculation
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # 4. HUD rendering
            output_frame = visualizer.render(annotated, pred_class, confidence, sentence, fps)

            cv2.imshow("Vaani-ISL: Real-Time Interpreter", output_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                predictor.clear_sentence()
            elif key == ord('s'):
                tts.speak(predictor.get_sentence_text())

    finally:
        stream.stop()
        extractor.close()
        tts.stop()
        cv2.destroyAllWindows()
        print("Application stopped.")


if __name__ == "__main__":
    main()
