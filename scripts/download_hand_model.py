"""Download the official MediaPipe detector asset (not the ISL classifier)."""
import argparse
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/checkpoints/hand_landmarker.task")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", path)
    print(f"Saved Hand Landmarker detector to {path}")


if __name__ == "__main__":
    main()
