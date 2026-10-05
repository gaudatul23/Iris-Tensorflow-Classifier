"""Batch / single-sample inference using persisted scaler + Keras model."""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import tensorflow as tf

from iris_classifier.config import AppConfig, load_config
from iris_classifier.data import CLASS_NAMES, FEATURE_NAMES


class IrisPredictor:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        keras_path = self.config.paths.keras_model(self.config.project_root)
        scaler_path = self.config.paths.scaler(self.config.project_root)
        if not keras_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                "Model artifacts missing. Run training first: python -m iris_classifier.train"
            )
        self.model = tf.keras.models.load_model(keras_path)
        self.scaler = joblib.load(scaler_path)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(np.asarray(features, dtype="float32"))
        return self.model.predict(scaled, verbose=0)

    def predict(self, features: np.ndarray) -> list[dict]:
        probs = self.predict_proba(features)
        results = []
        for row in probs:
            idx = int(np.argmax(row))
            results.append(
                {
                    "class_index": idx,
                    "class_name": CLASS_NAMES[idx],
                    "probabilities": {
                        name: float(p) for name, p in zip(CLASS_NAMES, row)
                    },
                    "feature_order": list(FEATURE_NAMES),
                }
            )
        return results


@lru_cache(maxsize=1)
def get_predictor(config_path: str | None = None) -> IrisPredictor:
    return IrisPredictor(load_config(config_path) if config_path else load_config())
