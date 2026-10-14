"""Webcam-to-text for the supplied static real-life word image model."""
import argparse
import os
import sys
import time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from src.tensorflow_isl.word_images import WordImageClassifier
from src.tensorflow_isl.stability import StablePrediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/checkpoints/tf_words")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", help="Optional prerecorded pipeline test")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    print("Loading TensorFlow word model...", flush=True)
    classifier = WordImageClassifier(args.model)
    print(f"Loaded {len(classifier.classes)} word classes. Opening camera/video...", flush=True)
    if not classifier.metadata["acceptance"]["validation_target_met"]:
        print("Validation acceptance target was not met; word text output is disabled.", flush=True)
    cap = cv2.VideoCapture(args.video or args.camera)
    if not cap.isOpened():
        classifier.close()
        raise RuntimeError("Could not open camera/video; check camera permissions and index")
    stabilizer = StablePrediction(frames=5)
    words, durations = [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            label, confidence, _ = classifier.predict(frame)
            stable, changed = stabilizer.update(label)
            if stable and changed:
                words.append(stable)
            durations.append((time.perf_counter()-start)*1000)
            if args.headless:
                continue
            display = cv2.flip(frame, 1)
            status = stable or ("Hold the word pose steady" if label else "Not sure - please sign again")
            cv2.putText(display, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            if label:
                cv2.putText(display, f"Confidence {confidence:.2f}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
            cv2.putText(display, " ".join(words)[-55:], (15, display.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
            cv2.imshow("Vaani ISL - TensorFlow word poses", display)
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), 27):
                break
            if key == ord('c'):
                words.clear()
                stabilizer = StablePrediction(frames=5)
    finally:
        cap.release()
        classifier.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"Processed {len(durations)} frames; average pipeline latency {sum(durations)/max(1,len(durations)):.1f} ms; text: {' '.join(words)}")


if __name__ == "__main__":
    main()
