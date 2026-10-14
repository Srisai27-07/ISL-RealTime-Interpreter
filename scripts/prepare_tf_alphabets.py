"""Audit and cache hand landmarks; preserves the provider's three split folders."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.tensorflow_isl.audit import audit_dataset
from src.tensorflow_isl.features import HandExtractor, FEATURE_VERSION, FEATURE_DIM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="data/processed/tf_alphabets")
    parser.add_argument("--hand-model", default="models/checkpoints/hand_landmarker.task")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    rows, summary = audit_dataset(args.data, output)
    print(json.dumps(summary["status_counts"]), flush=True)
    if args.audit_only:
        return
    extractor = HandExtractor(args.hand_model)
    caches = output / "cache" / FEATURE_VERSION
    caches.mkdir(parents=True, exist_ok=True)
    split_rows = {s: [] for s in ("Training", "Validation", "Testing")}
    failures = []
    try:
        for i, row in enumerate(rows):
            if row["status"] != "keep":
                continue
            path = Path(row["path"])
            cache = caches / (row["pixel_hash"] + ".npy")
            if cache.exists():
                feature = np.load(cache)
                feature = feature if feature.shape == (FEATURE_DIM,) else None
            else:
                image = cv2.imread(str(path))
                if image is None:
                    feature = None
                else:
                    feature, _, _ = extractor.extract(image)
                np.save(cache, feature if feature is not None else np.array([], dtype=np.float32))
            if feature is None:
                failures.append({"path": str(path), "split": row["split"],
                                 "label": row["label"], "reason": "no_valid_hand_landmarks"})
                continue
            split_rows[row["split"]].append((feature, row["label"], row["path"]))
            if i % 250 == 0:
                print(f"Landmarks processed {i}/{len(rows)}", flush=True)
    finally:
        extractor.close()
    for split, values in split_rows.items():
        np.savez_compressed(output / (split.lower()+".npz"),
            X=np.stack([v[0] for v in values]) if values else np.empty((0, FEATURE_DIM), np.float32),
            labels=np.array([v[1] for v in values], dtype=str),
            paths=np.array([v[2] for v in values], dtype=str))
    summary["landmark_failures"] = failures
    summary["valid_landmark_counts"] = {s: len(v) for s, v in split_rows.items()}
    summary["feature_version"] = FEATURE_VERSION
    summary["feature_dim"] = FEATURE_DIM
    summary["valid_by_class"] = {s: {label: sum(v[1]==label for v in values)
        for label in sorted({v[1] for v in values})} for s, values in split_rows.items()}
    (output / "audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["valid_landmark_counts"]), flush=True)


if __name__ == "__main__":
    main()
