"""Unit tests that do not require trained artifacts."""

import numpy as np

from iris_classifier.config import load_config
from iris_classifier.data import CLASS_NAMES, FEATURE_NAMES, load_raw_iris, prepare_splits
from iris_classifier.model import build_model


def test_iris_shape_and_classes():
    x, y = load_raw_iris()
    assert x.shape == (150, 4)
    assert set(y.tolist()) == {0, 1, 2}
    assert len(FEATURE_NAMES) == 4
    assert len(CLASS_NAMES) == 3


def test_splits_are_stratified_and_scaled():
    config = load_config()
    splits = prepare_splits(config)
    assert splits.x_train.shape[1] == 4
    assert abs(float(splits.x_train.mean())) < 1e-5
    assert splits.x_test.shape[0] == 30


def test_model_output_shape():
    config = load_config()
    model = build_model(config.model)
    out = model(np.zeros((2, 4), dtype="float32"))
    assert tuple(out.shape) == (2, 3)
