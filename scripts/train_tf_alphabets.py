"""TensorFlow training on Training only; policy selection on Validation only."""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import NearestNeighbors
from src.tensorflow_isl.features import FEATURE_VERSION, FEATURE_DIM


def dataset(X, y, train=False):
    data = tf.data.Dataset.from_tensor_slices((X, y))
    if train:
        data = data.shuffle(len(X), seed=42, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return data.batch(64).with_options(options).prefetch(1)


def select_policy(p, y, distances):
    predicted = p.argmax(axis=1)
    correct = predicted == y
    max_distance = float(np.quantile(distances[correct], .99)) if correct.any() else 0.0
    candidates = []
    for threshold in np.arange(.50, 1.00, .025):
        for margin in (0.10, 0.20, 0.30):
            accepted = ((p.max(axis=1) >= threshold) &
                        ((np.sort(p, axis=1)[:, -1]-np.sort(p, axis=1)[:, -2]) >= margin) &
                        (distances <= max_distance))
            n = int(accepted.sum())
            if n >= min(30, len(y)) and float(correct[accepted].mean()) >= .95:
                candidates.append((n, float(correct[accepted].mean()), float(threshold), margin))
    if not candidates:
        return {"threshold": 1.01, "margin": .30, "max_distance": max_distance,
                "validation_target_met": False, "note": "No policy achieved 95% accepted validation accuracy; live output disabled."}
    n, accuracy, threshold, margin = max(candidates)
    return {"threshold": threshold, "margin": margin, "max_distance": max_distance,
            "validation_target_met": True, "validation_coverage": n/len(y),
            "validation_accepted_accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/tf_alphabets")
    parser.add_argument("--output", default="models/checkpoints/tf_alphabets")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(42)
    base, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {s: np.load(base/(s.lower()+".npz")) for s in ("Training", "Validation", "Testing")}
    classes = sorted(set(arrays["Training"]["labels"]))
    if len(classes) != 26:
        raise ValueError(f"Expected all 26 alphabet classes in training, got {classes}")
    lookup = {v: i for i, v in enumerate(classes)}
    X = {s: d["X"] for s, d in arrays.items()}
    y = {s: np.array([lookup[v] for v in d["labels"]]) for s, d in arrays.items()}
    for split in arrays:
        counts = np.bincount(y[split], minlength=len(classes))
        if not len(y[split]) or counts.min() < 2:
            raise ValueError(f"Insufficient usable data in {split}: {dict(zip(classes, counts.tolist()))}")
    options = [(128, 64, .20), (256, 128, .30)]
    histories, best_loss, best_model = {}, float("inf"), None
    for index, (first, second, dropout) in enumerate(options):
        normalization = tf.keras.layers.Normalization()
        normalization.adapt(X["Training"])
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(FEATURE_DIM,)), normalization,
            tf.keras.layers.Dense(first, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(second, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(len(classes), activation="softmax")])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        counts = np.bincount(y["Training"], minlength=len(classes))
        weights = {i: len(y["Training"])/(len(classes)*int(c)) for i, c in enumerate(counts)}
        history = model.fit(dataset(X["Training"], y["Training"], True),
            validation_data=dataset(X["Validation"], y["Validation"]),
            epochs=args.epochs, class_weight=weights, verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
                       tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=.5),
                       tf.keras.callbacks.ModelCheckpoint(str(out/f"candidate_{index}.keras"), monitor="val_loss", save_best_only=True)])
        histories[str(index)] = history.history
        loss = float(model.evaluate(dataset(X["Validation"], y["Validation"]), verbose=0)[0])
        if loss < best_loss:
            best_loss, best_model = loss, index
    model = tf.keras.models.load_model(out/f"candidate_{best_model}.keras")
    model.save(out/"alphabet.keras")
    vp = model.predict(dataset(X["Validation"], y["Validation"]), verbose=0)
    tp = model.predict(dataset(X["Testing"], y["Testing"]), verbose=0)
    # Distance gate uses the predicted class and only training reference examples.
    neighbors = {i: NearestNeighbors(n_neighbors=1).fit(X["Training"][y["Training"]==i]) for i in range(len(classes))}
    def distances(features, p):
        result = np.empty(len(features))
        for i in range(len(classes)):
            mask = p.argmax(axis=1) == i
            if mask.any():
                result[mask] = neighbors[i].kneighbors(features[mask])[0][:, 0]
        return result
    vd, td = distances(X["Validation"], vp), distances(X["Testing"], tp)
    policy = select_policy(vp, y["Validation"], vd)
    allowed = {label: sorted(set(X["Training"][y["Training"]==i, 126:].sum(axis=1).astype(int).tolist()))
               for i, label in enumerate(classes)}
    # Include the same hand-count guard used by live inference in reported coverage.
    validation_accepted = ((vp.max(axis=1)>=policy["threshold"]) &
        (np.sort(vp,axis=1)[:,-1]-np.sort(vp,axis=1)[:,-2]>=policy["margin"]) &
        (vd<=policy["max_distance"]) & np.array([
            int(f[126:].sum()) in allowed[classes[i]]
            for f, i in zip(X["Validation"], vp.argmax(axis=1))]))
    policy["validation_coverage"] = float(validation_accepted.mean())
    policy["validation_accepted_accuracy"] = float((vp.argmax(axis=1)[validation_accepted]
        ==y["Validation"][validation_accepted]).mean()) if validation_accepted.any() else None
    np.savez_compressed(out/"reference.npz", X=X["Training"], y=y["Training"])
    metadata = {"classes": classes, "feature_version": FEATURE_VERSION, "feature_dim": FEATURE_DIM,
        "acceptance": policy, "allowed_hand_counts": allowed, "tensorflow_version": tf.__version__,
        "selected_candidate": best_model, "scope": "Static A-Z, one or two hands, text output",
        "mirroring": "No pixel mirroring before inference; mirrored preview only",
        "signer_independent_test_verified": False}
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2))
    accepted = ((tp.max(axis=1)>=policy["threshold"]) &
                (np.sort(tp,axis=1)[:,-1]-np.sort(tp,axis=1)[:,-2]>=policy["margin"]) &
                (td<=policy["max_distance"]) & np.array([
                    int(f[126:].sum()) in allowed[classes[i]]
                    for f, i in zip(X["Testing"], tp.argmax(axis=1))]))
    predicted = tp.argmax(axis=1)
    report = {"test_accuracy": float(accuracy_score(y["Testing"], predicted)),
        "classification_report": classification_report(y["Testing"], predicted, labels=list(range(26)), target_names=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y["Testing"], predicted, labels=list(range(26))).tolist(),
        "test_coverage": float(accepted.mean()),
        "test_accepted_accuracy": float((predicted[accepted]==y["Testing"][accepted]).mean()) if accepted.any() else None,
        "valid_samples": {s: len(values) for s, values in y.items()},
        "acceptance": policy, "signer_independent_test_verified": False,
        "unsupported_gesture_false_acceptance_tested": False,
        "webcam_validation_completed": False}
    report["test_acceptance_by_class"] = {
        label: {"samples": int((y["Testing"]==i).sum()),
                "accepted": int((accepted & (y["Testing"]==i)).sum()),
                "coverage": float(accepted[y["Testing"]==i].mean()),
                "accepted_accuracy": float((predicted[accepted & (y["Testing"]==i)]==i).mean())
                    if (accepted & (y["Testing"]==i)).any() else None}
        for i, label in enumerate(classes)}
    audit = json.loads((base/"audit.json").read_text())
    report["input_images"] = audit["total_images"]
    report["audit_status_counts"] = audit["status_counts"]
    report["test_landmark_detection_coverage"] = len(y["Testing"])/audit["retained_split_counts"]["Testing"]
    report["test_accepted_coverage_of_retained_images"] = int(accepted.sum())/audit["retained_split_counts"]["Testing"]
    (out/"evaluation.json").write_text(json.dumps(report, indent=2))
    (out/"history.json").write_text(json.dumps(histories, indent=2))
    for index in range(len(options)):
        (out/f"candidate_{index}.keras").unlink()
    print(json.dumps({k:v for k,v in report.items() if k not in ("classification_report", "confusion_matrix")}), flush=True)


if __name__ == "__main__":
    main()
