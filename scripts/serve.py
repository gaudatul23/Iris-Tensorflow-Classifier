"""Start the FastAPI process."""

import uvicorn

from iris_classifier.config import load_config

if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run("iris_classifier.api:app", host=cfg.api.host, port=cfg.api.port)
