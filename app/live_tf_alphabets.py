"""Webcam-to-text using the trained TensorFlow alphabet model and shared features."""
import argparse
import os
import sys
import time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from src.tensorflow_isl.features import HandExtractor
from src.tensorflow_isl.classifier import AlphabetClassifier
from src.tensorflow_isl.stability import StablePrediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/checkpoints/tf_alphabets")
    parser.add_argument("--hand-model", default="models/checkpoints/hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", help="Optional prerecorded video for integration testing")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    classifier = AlphabetClassifier(args.model)
    extractor = HandExtractor(args.hand_model, video=True)
    cap = cv2.VideoCapture(args.video or args.camera)
    if not cap.isOpened():
        extractor.close()
        raise RuntimeError("Could not open camera/video. Check camera permissions and device index.")
    stabilizer = StablePrediction(frames=5)
    text = ""
    frames, durations = 0, []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            feature, hands, square = extractor.extract(frame, int(time.monotonic()*1000))
            label, confidence, reason = classifier.predict(feature)
            stable, changed = stabilizer.update(label)
            if stable and changed:
                text += stable
            status = stable or ("Hold the sign steady" if label else
                                "No hand detected" if reason == "no_hand" else "Not sure - please sign again")
            durations.append((time.perf_counter()-start)*1000)
            frames += 1
            if args.headless:
                continue
            # Inference always sees the unmirrored original. Mirror only this preview.
            display = cv2.flip(square, 1)
            for hand in hands:
                for x, y, _ in hand:
                    cv2.circle(display, (int((1-x)*display.shape[1]), int(y*display.shape[0])), 3, (0,220,120), -1)
            cv2.putText(display, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            if label:
                cv2.putText(display, f"Confidence {confidence:.2f}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
            cv2.putText(display, text[-35:], (15, display.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2)
            cv2.imshow("Vaani ISL - TensorFlow alphabets", display)
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), 27):
                break
            if key == ord('c'):
                text = ""
    finally:
        cap.release()
        extractor.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"Processed {frames} frames; average pipeline latency {sum(durations)/max(1,len(durations)):.1f} ms; text: {text}")


if __name__ == "__main__":
    main()
