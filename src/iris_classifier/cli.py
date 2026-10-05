"""Console entrypoints."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from iris_classifier.config import load_config
from iris_classifier.evaluate import evaluate
from iris_classifier.train import train


def train_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the Iris MLP")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args(argv)
    train(load_config(args.config))


def evaluate_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained Iris MLP")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    evaluate(args.config)


def serve_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the Iris MLP via FastAPI")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_config()
    uvicorn.run(
        "iris_classifier.api:app",
        host=args.host or config.api.host,
        port=args.port or config.api.port,
        log_level="info",
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else "train"
    rest = argv[1:] if argv else []
    dispatch = {"train": train_cli, "evaluate": evaluate_cli, "serve": serve_cli}
    if command not in dispatch:
        raise SystemExit(f"Unknown command {command}. Use train | evaluate | serve")
    dispatch[command](rest)


if __name__ == "__main__":
    main()
