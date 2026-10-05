"""Configuration loading and path resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class DataConfig:
    test_size: float = 0.2
    val_size: float = 0.2
    batch_size: int = 16


@dataclass
class ModelConfig:
    hidden_units: list[int] = field(default_factory=lambda: [16, 8])
    dropout: float = 0.2
    learning_rate: float = 0.01


@dataclass
class TrainingConfig:
    epochs: int = 80
    patience: int = 12


@dataclass
class PathConfig:
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs/tensorboard"
    model_name: str = "iris_mlp"
    scaler_name: str = "scaler.joblib"
    metadata_name: str = "metadata.json"

    def artifacts(self, root: Path) -> Path:
        return (root / self.artifacts_dir).resolve()

    def logs(self, root: Path) -> Path:
        return (root / self.logs_dir).resolve()

    def saved_model(self, root: Path) -> Path:
        return self.artifacts(root) / "saved_model"

    def keras_model(self, root: Path) -> Path:
        return self.artifacts(root) / f"{self.model_name}.keras"

    def scaler(self, root: Path) -> Path:
        return self.artifacts(root) / self.scaler_name

    def metadata(self, root: Path) -> Path:
        return self.artifacts(root) / self.metadata_name


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    model_dir: str = "artifacts/saved_model"


@dataclass
class AppConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    project_root: Path = PROJECT_ROOT


def _from_dict(cls, data: dict[str, Any] | None):
    if not data:
        return cls()
    fields = cls.__dataclass_fields__
    return cls(**{k: v for k, v in data.items() if k in fields})


def load_config(path: Path | str | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    return AppConfig(
        seed=int(raw.get("seed", 42)),
        data=_from_dict(DataConfig, raw.get("data")),
        model=_from_dict(ModelConfig, raw.get("model")),
        training=_from_dict(TrainingConfig, raw.get("training")),
        paths=_from_dict(PathConfig, raw.get("paths")),
        api=_from_dict(ApiConfig, raw.get("api")),
        project_root=PROJECT_ROOT,
    )
