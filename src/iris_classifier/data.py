"""Iris dataset loading, splitting, scaling, and tf.data pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from iris_classifier.config import AppConfig

FEATURE_NAMES = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)
CLASS_NAMES = ("setosa", "versicolor", "virginica")


@dataclass
class IrisSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def load_raw_iris() -> tuple[np.ndarray, np.ndarray]:
    bunch = load_iris()
    return bunch.data.astype("float32"), bunch.target.astype("int32")


def prepare_splits(config: AppConfig) -> IrisSplits:
    features, labels = load_raw_iris()
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=config.data.test_size,
        random_state=config.seed,
        stratify=labels,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=config.data.val_size,
        random_state=config.seed,
        stratify=y_train_val,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_val = scaler.transform(x_val).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")
    return IrisSplits(x_train, y_train, x_val, y_val, x_test, y_test, scaler)


def make_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
