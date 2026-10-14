"""Load the Keras model and enforce the validation-selected acceptance policy."""
import json
from pathlib import Path
import numpy as np
from .features import FEATURE_VERSION, FEATURE_DIM


class AlphabetClassifier:
    def __init__(self, folder):
        import tensorflow as tf
        folder = Path(folder)
        self.metadata = json.loads((folder / "metadata.json").read_text())
        if self.metadata["feature_version"] != FEATURE_VERSION:
            raise ValueError("Model preprocessing version mismatch")
        self.classes = self.metadata["classes"]
        self.model = tf.keras.models.load_model(folder / "alphabet.keras", compile=False)
        with np.load(folder / "reference.npz") as reference:
            reference_x, reference_y = reference["X"], reference["y"]
        if (self.model.input_shape[-1] != FEATURE_DIM or
                self.model.output_shape[-1] != len(self.classes) or len(self.classes) < 2):
            raise ValueError("Model dimensions and class metadata do not match")
        if (reference_x.shape[1] != FEATURE_DIM or len(reference_x) != len(reference_y)):
            raise ValueError("Invalid training-reference features")
        self.references = {i: reference_x[reference_y == i] for i in range(len(self.classes))}

    def predict(self, feature):
        if feature is None:
            return None, 0.0, "no_hand"
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape != (FEATURE_DIM,) or not np.isfinite(feature).all():
            return None, 0.0, "invalid_landmarks"
        if feature[126:].sum() < 1:
            return None, 0.0, "no_hand"
        p = self.model(feature[None], training=False).numpy()[0]
        index = int(np.argmax(p))
        confidence = float(p[index])
        margin = confidence - float(np.partition(p, -2)[-2])
        hand_count = int(feature[126:].sum())
        expected = self.metadata["allowed_hand_counts"].get(self.classes[index], [])
        if expected and hand_count not in expected:
            return None, confidence, "incomplete_hand_detection"
        policy = self.metadata["acceptance"]
        if confidence < policy["threshold"] or margin < policy["margin"]:
            return None, confidence, "uncertain"
        ref = self.references[index]
        distance = float(np.min(np.linalg.norm(ref-feature, axis=1)))
        if distance > policy["max_distance"]:
            return None, confidence, "outside_training_distribution"
        return self.classes[index], confidence, "accepted"
