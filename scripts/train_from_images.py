"""
Direct Dataset Trainer: Extracts MediaPipe landmarks from Training/ images
and trains a high-accuracy Random Forest static sign classifier.

Run from the ISL project root:
    python3 scripts/train_from_images.py
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

sys.path.append(".")
from src.vision.landmark_extractor import ISLLandmarkExtractor

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_DIR    = "data/Dataset/Training"
TEST_DIR     = "data/Dataset/Testing"
MODEL_OUT    = "models/checkpoints/isl_static.pkl"
IMG_EXTS     = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
# ───────────────────────────────────────────────────────────────────────────────

def extract_features(data_dir: str, extractor: ISLLandmarkExtractor, label: str = ""):
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ])

    X, y = [], []
    skipped = 0

    for class_name in classes:
        class_folder = os.path.join(data_dir, class_name)
        img_paths = []
        for ext in IMG_EXTS:
            img_paths.extend(glob.glob(os.path.join(class_folder, ext)))
            img_paths.extend(glob.glob(os.path.join(class_folder, ext.upper())))

        for img_path in tqdm(img_paths, desc=f"  [{label}] {class_name}", leave=False):
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            features, _ = extractor.process_frame(img)

            # Skip blank / no-hand frames
            if np.all(features == 0):
                skipped += 1
                continue

            X.append(features)
            y.append(class_name)

    print(f"  Extracted {len(X)} valid samples | Skipped: {skipped}")
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    print("=" * 65)
    print("  Vaani-ISL Static Sign Trainer")
    print(f"  Training Dir  : {TRAIN_DIR}")
    print(f"  Testing Dir   : {TEST_DIR}")
    print(f"  Output Model  : {MODEL_OUT}")
    print("=" * 65)

    if not os.path.exists(TRAIN_DIR):
        print(f"[ERROR] Training directory not found: {TRAIN_DIR}")
        return

    extractor = ISLLandmarkExtractor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ── 1. Extract Training Landmarks ──────────────────────────────────────────
    print("\n[STEP 1/3] Extracting landmarks from Training images...")
    t0 = time.time()
    X_train, y_train = extract_features(TRAIN_DIR, extractor, label="TRAIN")
    print(f"  Done in {time.time() - t0:.1f}s | Shape: {X_train.shape}")

    if len(X_train) == 0:
        print("[ERROR] No hand landmarks extracted. Check dataset images.")
        extractor.close()
        return

    # ── 2. Extract Testing Landmarks ───────────────────────────────────────────
    X_test, y_test = np.array([]), np.array([])
    if os.path.exists(TEST_DIR):
        print("\n[STEP 2/3] Extracting landmarks from Testing images...")
        t0 = time.time()
        X_test, y_test = extract_features(TEST_DIR, extractor, label="TEST")
        print(f"  Done in {time.time() - t0:.1f}s | Shape: {X_test.shape}")
    else:
        print("\n[STEP 2/3] No separate Testing dir found. Will use 20% auto-split.")

    extractor.close()
    classes = sorted(list(np.unique(y_train)))
    print(f"\n  Classes ({len(classes)}): {classes}")

    # ── 3. Train Random Forest Classifier ─────────────────────────────────────
    print("\n[STEP 3/3] Training Random Forest Classifier...")
    print("  (n_estimators=200, max_depth=30, n_jobs=-1 — uses all CPU cores)")
    t0 = time.time()

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1         # Use ALL CPU cores
    )
    clf.fit(X_train, y_train)
    print(f"  Training completed in {time.time() - t0:.1f}s")

    # ── 4. Evaluate ────────────────────────────────────────────────────────────
    print("\n─── EVALUATION REPORT ───────────────────────────────────────")
    if len(X_test) > 0:
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n  🎯 Test Accuracy  : {acc * 100:.2f}%")
        print(f"\n{classification_report(y_test, y_pred, target_names=classes)}")
    else:
        train_pred = clf.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"\n  Training Accuracy : {train_acc * 100:.2f}%")
        print("  (Run with a Testing dir for proper held-out accuracy)")

    # ── 5. Save Model ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    payload = {
        "model":        clf,
        "classes":      classes,
        "num_features": X_train.shape[1]
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(payload, f)

    size_mb = os.path.getsize(MODEL_OUT) / (1024 * 1024)
    print(f"\n  ✅  Model saved  → {MODEL_OUT}  ({size_mb:.1f} MB)")
    print("  Your FastAPI server will auto-reload and go LIVE immediately!")
    print("=" * 65)


if __name__ == "__main__":
    main()
