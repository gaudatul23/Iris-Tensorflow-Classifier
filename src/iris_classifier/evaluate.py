"""Offline evaluation of a trained Keras / SavedModel artifact."""

from __future__ import annotations

import json

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from iris_classifier.config import load_config
from iris_classifier.data import CLASS_NAMES, load_raw_iris
from iris_classifier.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    keras_path = config.paths.keras_model(config.project_root)
    if not keras_path.exists():
        raise FileNotFoundError(f"Missing model at {keras_path}. Train first.")
    model = tf.keras.models.load_model(keras_path)
    scaler = joblib.load(config.paths.scaler(config.project_root))

    features, labels = load_raw_iris()
    _, x_test, _, y_test = train_test_split(
        features,
        labels,
        test_size=config.data.test_size,
        random_state=config.seed,
        stratify=labels,
    )
    x_test = scaler.transform(x_test).astype("float32")
    probs = model.predict(x_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    report = classification_report(
        y_test, preds, target_names=CLASS_NAMES, output_dict=True
    )
    cm = confusion_matrix(y_test, preds).tolist()
    result = {"classification_report": report, "confusion_matrix": cm}
    out = config.paths.artifacts(config.project_root) / "eval_report.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info("\n%s", classification_report(y_test, preds, target_names=CLASS_NAMES))
    return result


if __name__ == "__main__":
    evaluate()
