"""Frozen ImageNet image embeddings + TensorFlow head, with held-out User groups."""
import argparse
import json
from importlib.metadata import version
import os
import sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from src.tensorflow_isl.word_images import prepare_rgb, WORD_FEATURE_VERSION, WORD_EMBEDDING_VERSION, WORD_IMAGE_SIZE
from src.tensorflow_isl.features import FEATURE_VERSION, HandExtractor
from scripts.train_tf_alphabets import select_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/tf_words")
    parser.add_argument("--output", default="models/checkpoints/tf_words")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hand-model", default="models/checkpoints/hand_landmarker.task")
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(42)
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((data / "audit.json").read_text())
    if not audit.get("user_disjoint_split"):
        raise ValueError("Run prepare_tf_words.py to establish fixed User-group splits")
    rows = [r for r in json.loads((data / "manifest.json").read_text()) if r["status"] == "keep"]
    classes = sorted({r["label"] for r in rows if r["split"] == "Training"})
    if len(classes) != 20:
        raise ValueError(f"Expected twenty supplied word labels, got {classes}")
    lookup = {c: i for i, c in enumerate(classes)}
    backbone = tf.keras.applications.MobileNetV2(input_shape=(WORD_IMAGE_SIZE, WORD_IMAGE_SIZE, 3),
        alpha=.35, include_top=False, weights="imagenet", pooling="avg")
    backbone.trainable = False
    cache = data / "image_embeddings" / WORD_EMBEDDING_VERSION
    cache.mkdir(parents=True, exist_ok=True)
    X, H, y, paths, image_failures = {}, {}, {}, {}, []
    hand_cache = data / "cache" / FEATURE_VERSION
    hand_cache.mkdir(parents=True, exist_ok=True)
    hand_extractor = None
    for split in ("Training", "Validation", "Testing"):
        values = []
        pending = []
        def flush():
            if not pending:
                return
            images = np.stack([v[0] for v in pending])
            features = backbone(images / 127.5 - 1., training=False).numpy()
            for (_, row, destination), feature in zip(pending, features):
                np.save(destination, feature)
                values.append((feature, lookup[row["label"]], row["path"]))
            pending.clear()
        split_rows = [r for r in rows if r["split"] == split]
        for i, row in enumerate(split_rows):
            destination = cache / (row["pixel_hash"] + ".npy")
            if destination.exists():
                values.append((np.load(destination), lookup[row["label"]], row["path"]))
            else:
                try:
                    with Image.open(row["path"]) as image:
                        pixels = prepare_rgb(image)
                except Exception as error:
                    image_failures.append({"path": row["path"], "reason": str(error)})
                    continue
                pending.append((pixels, row, destination))
                if len(pending) == 32:
                    flush()
            if i % 256 == 0:
                print(f"{split}: image embeddings {i}/{len(split_rows)}", flush=True)
        flush()
        X[split] = np.stack([v[0] for v in values])
        by_path = {r["path"]: r for r in split_rows}
        hands = []
        for _, _, path in values:
            row = by_path[path]
            hp = hand_cache / (row["pixel_hash"] + ".npy")
            if hp.exists():
                feature = np.load(hp)
            else:
                import cv2
                if hand_extractor is None:
                    hand_extractor = HandExtractor(args.hand_model)
                feature, _, _ = hand_extractor.extract(cv2.imread(path))
                feature = feature if feature is not None else np.array([], dtype=np.float32)
                np.save(hp, feature)
            hands.append(feature if feature.shape == (128,) else np.zeros(128, dtype=np.float32))
        H[split] = np.stack(hands)
        y[split] = np.array([v[1] for v in values])
        paths[split] = np.array([v[2] for v in values])
        counts = np.bincount(y[split], minlength=len(classes))
        if counts.min() < 2:
            raise ValueError(f"Insufficient distinct images in {split}: {dict(zip(classes, counts.tolist()))}")
        np.savez_compressed(data / (split.lower()+"_images.npz"), X=X[split], y=y[split], paths=paths[split])
    if hand_extractor is not None:
        hand_extractor.close()
    def hybrid_data(split, train=False):
        values = tf.data.Dataset.from_tensor_slices(({"image_embedding": X[split], "hand_features": H[split]}, y[split]))
        if train:
            values = values.shuffle(len(y[split]), seed=42, reshuffle_each_iteration=True)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        return values.batch(64).with_options(options).prefetch(1)
    histories, best_loss, selected = {}, float("inf"), None
    for index, (hidden, dropout, image_units) in enumerate(((128, .25, 8), (256, .35, 32))):
        tf.keras.utils.set_random_seed(42 + index)
        # Embeddings are fixed; only the classification head is fitted.
        hi = tf.keras.Input(shape=(128,), name="hand_features")
        ei = tf.keras.Input(shape=(1280,), name="image_embedding")
        norm = tf.keras.layers.Normalization()
        norm.adapt(H["Training"])
        hand_branch = tf.keras.layers.Dense(hidden, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(norm(hi))
        image_branch = tf.keras.layers.Dense(image_units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(.01))(tf.keras.layers.UnitNormalization(axis=-1)(ei))
        image_branch = tf.keras.layers.Dropout(.5)(image_branch)
        combined = tf.keras.layers.Concatenate()([hand_branch, image_branch])
        combined = tf.keras.layers.Dropout(dropout)(combined)
        combined = tf.keras.layers.Dense(hidden//2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(combined)
        probabilities = tf.keras.layers.Dense(len(classes), activation="softmax")(combined)
        head = tf.keras.Model({"image_embedding": ei, "hand_features": hi}, probabilities)
        head.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        counts = np.bincount(y["Training"], minlength=len(classes))
        weights = {i: len(y["Training"])/(len(classes)*int(c)) for i, c in enumerate(counts)}
        h = head.fit(hybrid_data("Training", True),
            validation_data=hybrid_data("Validation"), epochs=args.epochs,
            class_weight=weights, verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
                       tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=.5)])
        histories[str(index)] = h.history
        loss = float(head.evaluate(hybrid_data("Validation"), verbose=0)[0])
        if loss < best_loss:
            best_loss, selected = loss, index
            head.save(out / "head.keras")
    head = tf.keras.models.load_model(out / "head.keras", compile=False)
    vp = head({"image_embedding": X["Validation"], "hand_features": H["Validation"]}, training=False).numpy()
    distance_features = {s: np.concatenate([H[s], X[s] / np.maximum(np.linalg.norm(X[s], axis=1, keepdims=True), 1e-8)], axis=1) for s in X}
    neighbors = {i: NearestNeighbors(n_neighbors=1).fit(distance_features["Training"][y["Training"] == i]) for i in range(len(classes))}
    def distances(features, p):
        result = np.empty(len(features))
        for i in range(len(classes)):
            mask = p.argmax(axis=1) == i
            if mask.any():
                result[mask] = neighbors[i].kneighbors(features[mask])[0][:, 0]
        return result
    policy = select_policy(vp, y["Validation"], distances(distance_features["Validation"], vp))
    # Freeze architecture and acceptance policy before touching test predictions.
    inputs = tf.keras.Input(shape=(WORD_IMAGE_SIZE, WORD_IMAGE_SIZE, 3), name="rgb_image")
    hi = tf.keras.Input(shape=(128,), name="hand_features")
    embedding = backbone(tf.keras.layers.Rescaling(1./127.5, offset=-1.)(inputs), training=False)
    df = tf.keras.layers.Concatenate()([hi, tf.keras.layers.UnitNormalization(axis=-1)(embedding)])
    model = tf.keras.Model({"rgb_image": inputs, "hand_features": hi},
        {"probabilities": head({"image_embedding": embedding, "hand_features": hi}), "distance_features": df})
    model.save(out / "word.keras")
    (out / "head.keras").unlink()
    np.savez_compressed(out / "reference.npz", X=distance_features["Training"], y=y["Training"])
    metadata = {"classes": classes, "feature_version": WORD_FEATURE_VERSION,
        "input_size": WORD_IMAGE_SIZE, "acceptance": policy, "selected_candidate": selected,
        "scope": "Twenty supplied still-image word poses; one or two hands",
        "training_method": "Frozen ImageNet MobileNetV2 image features plus optional spatial hand landmarks, trained TensorFlow fusion head",
        "split_users": audit["split_users"], "provider_user_disjoint_split": True,
        "signer_independent_test_verified": False, "motion_recognition_trained": False,
        "no_hand_rejection_guaranteed": False, "tensorflow_version": tf.__version__}
    metadata["mediapipe_version"] = version("mediapipe")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    tp = head({"image_embedding": X["Testing"], "hand_features": H["Testing"]}, training=False).numpy()
    td = distances(distance_features["Testing"], tp)
    accepted = ((tp.max(axis=1) >= policy["threshold"]) &
        (np.sort(tp, axis=1)[:, -1]-np.sort(tp, axis=1)[:, -2] >= policy["margin"]) &
        (td <= policy["max_distance"]))
    predicted = tp.argmax(axis=1)
    report = {"test_accuracy": float(accuracy_score(y["Testing"], predicted)),
        "classification_report": classification_report(y["Testing"], predicted, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y["Testing"], predicted, labels=list(range(len(classes)))).tolist(),
        "test_accepted_accuracy": float((predicted[accepted] == y["Testing"][accepted]).mean()) if accepted.any() else None,
        "test_coverage": float(accepted.mean()), "test_accepted_count": int(accepted.sum()),
        "valid_images": {s: len(v) for s, v in y.items()}, "input_images": audit["total_images"],
        "audit_status_counts": audit["status_counts"], "image_failures": image_failures,
        "acceptance": policy, "provider_user_disjoint_split": True,
        "signer_independent_test_verified": False, "webcam_validation_completed": False,
        "unsupported_gesture_false_acceptance_tested": False,
        "no_hand_rejection_guaranteed": False}
    report["test_group_evaluated_in_prior_image_only_prototype"] = True
    report["test_acceptance_by_class"] = {label: {
        "samples": int((y["Testing"] == i).sum()), "accepted": int((accepted & (y["Testing"] == i)).sum()),
        "coverage": float(accepted[y["Testing"] == i].mean()),
        "accepted_accuracy": float((predicted[accepted & (y["Testing"] == i)] == i).mean())
            if (accepted & (y["Testing"] == i)).any() else None}
        for i, label in enumerate(classes)}
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    (out / "history.json").write_text(json.dumps(histories, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("classification_report", "confusion_matrix", "test_acceptance_by_class")}), flush=True)


if __name__ == "__main__":
    main()
