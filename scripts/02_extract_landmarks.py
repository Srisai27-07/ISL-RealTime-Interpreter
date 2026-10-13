"""
Offline Batch Landmark Extractor.
Converts raw video files (.mp4, .avi, .mov) from public datasets (e.g. INCLUDE, ISL-CSLTR)
into standardized 30-frame landmark sequences (.npy) for model training.

Usage:
    python scripts/02_extract_landmarks.py --input-dir data/raw --output-dir data/processed
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(".")

from src.vision.landmark_extractor import ISLLandmarkExtractor


def extract_landmarks_from_video(
    video_path: str,
    extractor: ISLLandmarkExtractor,
    target_frames: int = 30
) -> np.ndarray:
    """Reads a video file and produces a normalized (target_frames, feature_dim) array."""
    cap = cv2.VideoCapture(video_path)
    frames_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        features, _ = extractor.process_frame(frame)
        frames_features.append(features)

    cap.release()

    if len(frames_features) == 0:
        return np.zeros((target_frames, 126), dtype=np.float32)

    # Resample to exactly target_frames (e.g. 30 frames)
    indices = np.linspace(0, len(frames_features) - 1, target_frames).astype(int)
    resampled = [frames_features[idx] for idx in indices]
    return np.array(resampled, dtype=np.float32)


def process_dataset_directory(input_dir: str, output_dir: str, target_frames: int = 30):
    os.makedirs(output_dir, exist_ok=True)
    extractor = ISLLandmarkExtractor()

    # Supports both flat folders and class-named subfolders
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    if not all_videos:
        print(f"No video files found in {input_dir}. Place raw videos in subfolders by class name.")
        return

    print(f"Found {len(all_videos)} video files. Extracting landmarks...")

    for v_path in tqdm(all_videos, desc="Extracting landmarks"):
        class_name = os.path.basename(os.path.dirname(v_path))
        base_name = os.path.splitext(os.path.basename(v_path))[0]

        target_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)
        out_npy = os.path.join(target_class_dir, f"{base_name}.npy")

        if os.path.exists(out_npy):
            continue

        landmarks = extract_landmarks_from_video(v_path, extractor, target_frames=target_frames)
        np.save(out_npy, landmarks)

    extractor.close()
    print("Batch landmark extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch video to landmark extractor.")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Folder containing raw videos")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Folder to save .npy files")
    parser.add_argument("--frames", type=int, default=30, help="Target sequence length per video")
    args = parser.parse_args()

    process_dataset_directory(args.input_dir, args.output_dir, args.frames)
