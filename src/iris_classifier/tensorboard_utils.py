"""TensorBoard helpers: scalars, histograms, confusion matrix, embeddings."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from iris_classifier.data import CLASS_NAMES


def plot_confusion_matrix(cm: np.ndarray) -> tf.Tensor:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Validation confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    return tf.expand_dims(image, 0)


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val: np.ndarray, y_val: np.ndarray, log_dir: Path):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.writer = tf.summary.create_file_writer(str(log_dir / "cm"))

    def on_epoch_end(self, epoch, logs=None):
        preds = np.argmax(self.model.predict(self.x_val, verbose=0), axis=1)
        cm = confusion_matrix(self.y_val, preds, labels=[0, 1, 2])
        image = plot_confusion_matrix(cm)
        with self.writer.as_default():
            tf.summary.image("confusion_matrix", image, step=epoch)
            self.writer.flush()


def write_projector_metadata(log_dir: Path, x_val: np.ndarray, y_val: np.ndarray) -> None:
    """Write embedding + metadata so TensorBoard Projector can show validation samples."""
    projector_dir = log_dir / "projector"
    projector_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.train.Checkpoint(embedding=tf.Variable(x_val, name="iris_features"))
    checkpoint.save(str(projector_dir / "embedding.ckpt"))
    metadata_path = projector_dir / "metadata.tsv"
    with metadata_path.open("w", encoding="utf-8") as handle:
        handle.write("label\n")
        for label in y_val:
            handle.write(f"{CLASS_NAMES[int(label)]}\n")
    config_path = projector_dir / "projector_config.pbtxt"
    config_path.write_text(
        'embeddings {\n'
        '  tensor_name: "embedding/.ATTRIBUTES/VARIABLE_VALUE"\n'
        f'  metadata_path: "{metadata_path.as_posix()}"\n'
        "}\n",
        encoding="utf-8",
    )
