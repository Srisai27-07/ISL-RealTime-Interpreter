"""
Integrated ISL Vocabulary Trainer:
Extracts landmarks from both A-Z Alphabets AND Real-Life Words (doctor, pain, you, work, etc.)
Trains a unified, high-performance Random Forest classifier.

Run:
    python3 scripts/train_unified_isl.py
"""

import os
import sys
import glob
import pickle
import time
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append(".")
from src.vision.landmark_extractor import ISLLandmarkExtractor

ALPHABET_DIR = "data/Dataset/Training"
WORDS_DIR    = "data/Dataset/RealLifeWords"
MODEL_OUT    = "models/checkpoints/isl_static.pkl"
MAX_SAMPLES_PER_CLASS = 400  # Cap per class for balanced, ultra-fast multi-core training


def extract_features_from_folder(base_dir: str, extractor: ISLLandmarkExtractor, prefix: str = ""):
    classes = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    ])
    X, y = [], []

    for class_name in classes:
        class_folder = os.path.join(base_dir, class_name)
        img_paths = glob.glob(os.path.join(class_folder, "**/*.jpg"), recursive=True)
        if not img_paths:
            img_paths = glob.glob(os.path.join(class_folder, "*.jpg"))

        # Limit per class to balance dataset and maintain high training speed
        selected_paths = img_paths[:MAX_SAMPLES_PER_CLASS]
        label = f"{prefix}{class_name}"

        for img_path in tqdm(selected_paths, desc=f"  [EXTRACT] {label}", leave=False):
            img = cv2.imread(img_path)
            if img is None:
                continue

            features, _ = extractor.process_frame(img)
            # Skip frames where no hand is visible at all
            if np.all(features == 0):
                continue

            X.append(features)
            y.append(label)

    return X, y


def main():
    print("=" * 70)
    print("  Vaani-ISL Unified Vocabulary Trainer")
    print("  Including: 26 ISL Alphabets (A-Z) + 20 Real-Life ISL Words")
    print(f"  Target Checkpoint: {MODEL_OUT}")
    print("=" * 70)

    extractor = ISLLandmarkExtractor(
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45
    )

    t0 = time.time()
    X_all, y_all = [], []

    # 1. Extract Alphabets
    if os.path.exists(ALPHABET_DIR):
        print("\n[STEP 1/3] Extracting ISL Alphabets (A-Z)...")
        xa, ya = extract_features_from_folder(ALPHABET_DIR, extractor)
        X_all.extend(xa)
        y_all.extend(ya)
        print(f"  ✓ Alphabets extracted: {len(xa)} samples")

    # 2. Extract Real-Life Words
    if os.path.exists(WORDS_DIR):
        print("\n[STEP 2/3] Extracting Real-Life Words (doctor, pain, work, pray, etc.)...")
        xw, yw = extract_features_from_folder(WORDS_DIR, extractor)
        X_all.extend(xw)
        y_all.extend(yw)
        print(f"  ✓ Words extracted: {len(xw)} samples")

    extractor.close()

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all)
    classes = sorted(list(np.unique(y)))

    print(f"\n  Total Dataset: {len(X)} samples across {len(classes)} classes")
    print(f"  Classes: {classes}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # 3. Train High-Capacity Random Forest
    print("\n[STEP 3/3] Training Random Forest with All CPU Cores (n_jobs=-1)...")
    clf = RandomForestClassifier(
        n_estimators=220,
        max_depth=35,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    t_train = time.time()
    clf.fit(X_train, y_train)
    print(f"  Training finished in {time.time() - t_train:.1f}s")

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 70)
    print(f"  🎯 UNIFIED TEST ACCURACY: {acc * 100:.2f}%")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=classes))

    # Save
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    payload = {
        "model": clf,
        "classes": classes,
        "num_features": X.shape[1]
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(payload, f)

    size_mb = os.path.getsize(MODEL_OUT) / (1024 * 1024)
    print(f"\n  ✅ Model successfully saved to {MODEL_OUT} ({size_mb:.1f} MB)")
    print(f"  Vocabulary now supports {len(classes)} real signs!")
    print(f"  Total pipeline time: {time.time() - t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
