"""Identical RGB square-padding preprocessing for training and webcam word poses."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

WORD_IMAGE_SIZE = 160
WORD_EMBEDDING_VERSION = "rgb-square-mobilenetv2-035-160-v1"
WORD_FEATURE_VERSION = "rgb-square-and-spatial-hands-v2"


def prepare_rgb(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    side = max(image.size)
    square = Image.new("RGB", (side, side), (0, 0, 0))
    square.paste(image, ((side-image.width)//2, (side-image.height)//2))
    return np.asarray(square.resize((WORD_IMAGE_SIZE, WORD_IMAGE_SIZE),
                                   Image.Resampling.BILINEAR), dtype=np.float32)


def prepare_bgr(frame):
    return prepare_rgb(Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1])))


class WordImageClassifier:
    def __init__(self, folder):
        import tensorflow as tf
        folder = Path(folder)
        self.metadata = json.loads((folder / "metadata.json").read_text())
        if self.metadata["feature_version"] != WORD_FEATURE_VERSION:
            raise ValueError("Word model preprocessing mismatch")
        self.classes = self.metadata["classes"]
        self.model = tf.keras.models.load_model(folder / "word.keras", compile=False)
        from .features import HandExtractor
        self.extractor = HandExtractor(folder.parent / "hand_landmarker.task")
        with np.load(folder / "reference.npz") as reference:
            x, y = reference["X"], reference["y"]
        self.references = {i: x[y == i] for i in range(len(self.classes))}

    def predict(self, frame):
        if frame is None or frame.size == 0:
            return None, 0., "no_frame"
        hand, _, _ = self.extractor.extract(frame)
        hand = hand if hand is not None else np.zeros(128, dtype=np.float32)
        result = self.model({"rgb_image": prepare_bgr(frame)[None], "hand_features": hand[None]}, training=False)
        p = result["probabilities"].numpy()[0]
        embedding = result["distance_features"].numpy()[0]
        index = int(p.argmax())
        confidence = float(p[index])
        margin = confidence - float(np.partition(p, -2)[-2])
        policy = self.metadata["acceptance"]
        if confidence < policy["threshold"] or margin < policy["margin"]:
            return None, confidence, "uncertain"
        distance = float(np.min(np.linalg.norm(self.references[index]-embedding, axis=1)))
        if distance > policy["max_distance"]:
            return None, confidence, "outside_training_distribution"
        return self.classes[index], confidence, "accepted"

    def close(self):
        self.extractor.close()
