"""
Lightweight Static Sign Classifier for Single-Frame ISL Hand Shapes (Alphabet / Digits).
Uses Random Forest or Multi-Layer Perceptron (MLP) for zero-latency single-frame inference.
"""

import pickle
import os
from typing import Tuple, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class StaticSignClassifier:
    """Classifies single-frame 2-hand landmarks (126,) using Random Forest."""

    def __init__(self, n_estimators: int = 150):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.class_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, class_names: List[str]):
        """X: (N, 126), y: (N,) integers"""
        self.class_names = class_names
        self.clf.fit(X, y)

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        """
        Predicts class and confidence for a single frame feature vector (126,).
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        probas = self.clf.predict_proba(feature_vector)[0]
        max_idx = int(np.argmax(probas))
        confidence = float(probas[max_idx])
        label = self.class_names[max_idx]
        return label, confidence

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"model": self.clf, "classes": self.class_names}, f)

    @classmethod
    def load(cls, filepath: str) -> "StaticSignClassifier":
        instance = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        instance.clf = data["model"]
        instance.class_names = data["classes"]
        return instance
