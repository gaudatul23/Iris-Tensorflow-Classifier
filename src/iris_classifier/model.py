"""Keras MLP for 3-class Iris classification."""

from __future__ import annotations

import tensorflow as tf

from iris_classifier.config import ModelConfig


def build_model(config: ModelConfig, n_features: int = 4, n_classes: int = 3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(n_features,), name="features")
    x = inputs
    for i, units in enumerate(config.hidden_units):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = tf.keras.layers.Dropout(config.dropout, name=f"dropout_{i}")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="probabilities")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="iris_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )
    return model
