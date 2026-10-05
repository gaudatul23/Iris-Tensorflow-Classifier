"""Training entrypoint with checkpoints, early stopping, and TensorBoard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import tensorflow as tf

from iris_classifier import __version__
from iris_classifier.config import AppConfig, load_config
from iris_classifier.data import CLASS_NAMES, FEATURE_NAMES, make_dataset, prepare_splits
from iris_classifier.logging_utils import get_logger
from iris_classifier.model import build_model
from iris_classifier.tensorboard_utils import ConfusionMatrixCallback, write_projector_metadata

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)


def train(config: AppConfig | None = None) -> dict:
    config = config or load_config()
    set_seed(config.seed)
    splits = prepare_splits(config)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = config.paths.logs(config.project_root) / run_id
    artifacts = config.paths.artifacts(config.project_root)
    artifacts.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset(
        splits.x_train,
        splits.y_train,
        config.data.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_ds = make_dataset(
        splits.x_val,
        splits.y_val,
        config.data.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    model = build_model(config.model, n_features=splits.x_train.shape[1])
    keras_path = config.paths.keras_model(config.project_root)

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
            profile_batch=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.training.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(keras_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        ConfusionMatrixCallback(splits.x_val, splits.y_val, log_dir),
    ]

    logger.info("Starting training run_id=%s log_dir=%s", run_id, log_dir)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.training.epochs,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,
    )

    saved_model_dir = config.paths.saved_model(config.project_root)
    if hasattr(model, "export"):
        model.export(saved_model_dir)
    else:
        model.save(saved_model_dir, save_format="tf")
    joblib.dump(splits.scaler, config.paths.scaler(config.project_root))
    write_projector_metadata(log_dir, splits.x_val, splits.y_val)

    test_metrics = model.evaluate(
        splits.x_test,
        splits.y_test,
        verbose=0,
        return_dict=True,
    )
    metadata = {
        "version": __version__,
        "run_id": run_id,
        "feature_names": list(FEATURE_NAMES),
        "class_names": list(CLASS_NAMES),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "epochs_trained": len(history.history["loss"]),
        "tensorboard_log_dir": str(log_dir),
        "saved_model_dir": str(saved_model_dir),
    }
    config.paths.metadata(config.project_root).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    logger.info("Training complete: %s", metadata["test_metrics"])
    logger.info("TensorBoard: tensorboard --logdir %s", config.paths.logs(config.project_root))
    return metadata


def main(config_path: str | Path | None = None) -> dict:
    return train(load_config(config_path))


if __name__ == "__main__":
    main()
