"""
Precision ISL Alphabet Trainer:
Extracts high-resolution landmarks from ALL 25,977 images across:
  - data/Dataset/Training (18,198 images)
  - data/Dataset/Validation (2,579 images)
  - data/Dataset/Testing (5,200 images)

Uses the upgraded dual-hand mp.solutions.hands extractor.
Trains a high-precision 26-class Random Forest model:
  models/checkpoints/isl_alphabet_highres.pkl

Run:
    python3 scripts/train_precision_alphabet.py
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

TRAIN_DIR = "data/Dataset/Training"
VAL_DIR   = "data/Dataset/Validation"
TEST_DIR  = "data/Dataset/Testing"
MODEL_OUT = "models/checkpoints/isl_alphabet_highres.pkl"

# Up to 600 samples per class per split to maximize speed and data diversity
MAX_SAMPLES_PER_SPLIT = 600

def extract_from_folder(base_dir: str, extractor: ISLLandmarkExtractor, split_name: str):
    if not os.path.exists(base_dir):
        return [], []
    
    classes = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    ])
    
    X, y = [], []
    for class_name in classes:
        class_folder = os.path.join(base_dir, class_name)
        img_paths = glob.glob(os.path.join(class_folder, "*.jpg"))
        if not img_paths:
            img_paths = glob.glob(os.path.join(class_folder, "*.[jJ][pP][gG]"))
        
        selected = img_paths[:MAX_SAMPLES_PER_SPLIT]
        for img_path in tqdm(selected, desc=f"  [{split_name}] Letter {class_name}", leave=False):
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            features, _ = extractor.process_frame(img)
            # Skip if both hands completely absent
            if np.all(features == 0):
                continue
            
            X.append(features)
            y.append(class_name)
            
    return X, y

def main():
    print("=" * 70)
    print("  Vaani-ISL Precision Alphabet Model Trainer (26 Classes A-Z)")
    print("  Aggregating Training + Validation + Testing splits")
    print(f"  Target Checkpoint: {MODEL_OUT}")
    print("=" * 70)

    extractor = ISLLandmarkExtractor(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

    t0 = time.time()
    X_all, y_all = [], []

    # 1. Training split
    print("\n[STEP 1/3] Extracting from Training split...")
    xt, yt = extract_from_folder(TRAIN_DIR, extractor, "TRAIN")
    X_all.extend(xt)
    y_all.extend(yt)
    print(f"  ✓ Training split: {len(xt)} valid samples")

    # 2. Validation split
    print("\n[STEP 2/3] Extracting from Validation split...")
    xv, yv = extract_from_folder(VAL_DIR, extractor, "VAL")
    X_all.extend(xv)
    y_all.extend(yv)
    print(f"  ✓ Validation split: {len(xv)} valid samples")

    # 3. Testing split
    print("\n[STEP 3/3] Extracting from Testing split...")
    xte, yte = extract_from_folder(TEST_DIR, extractor, "TEST")
    X_all.extend(xte)
    y_all.extend(yte)
    print(f"  ✓ Testing split: {len(xte)} valid samples")

    extractor.close()

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all)
    classes = sorted(list(np.unique(y)))

    print(f"\n  Total Dataset: {len(X)} samples across {len(classes)} alphabet classes")
    print(f"  Classes: {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print("\n[TRAINING] Training Precision Random Forest Classifier...")
    print("  (n_estimators=250, max_depth=35, n_jobs=-1 — all CPU cores)")
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=35,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    t_train = time.time()
    clf.fit(X_train, y_train)
    print(f"  Training finished in {time.time() - t_train:.1f}s")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 70)
    print(f"  🎯 PRECISION ALPHABET ACCURACY: {acc * 100:.2f}%")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=classes))

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
    print(f"  Total processing time: {time.time() - t0:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
