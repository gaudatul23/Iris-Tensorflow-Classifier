"""FastAPI production serving layer."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from iris_classifier import __version__
from iris_classifier.data import FEATURE_NAMES
from iris_classifier.logging_utils import get_logger
from iris_classifier.predict import IrisPredictor

logger = get_logger(__name__)
predictor: IrisPredictor | None = None


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=15, examples=[5.1])
    sepal_width: float = Field(..., ge=0, le=15, examples=[3.5])
    petal_length: float = Field(..., ge=0, le=15, examples=[1.4])
    petal_width: float = Field(..., ge=0, le=15, examples=[0.2])

    def as_row(self) -> list[float]:
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class PredictRequest(BaseModel):
    samples: list[IrisFeatures] = Field(..., min_length=1, max_length=64)

    @model_validator(mode="after")
    def _non_empty(self):
        return self


class PredictResponse(BaseModel):
    predictions: list[dict]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = IrisPredictor()
        logger.info("Loaded IrisPredictor")
    except FileNotFoundError as exc:
        logger.warning("API started without model: %s", exc)
        predictor = None
    yield
    predictor = None


app = FastAPI(
    title="Iris Classifier API",
    version=__version__,
    description="Production FastAPI service for the TensorFlow Iris MLP.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok" if predictor is not None else "model_not_loaded",
        "version": __version__,
        "features": list(FEATURE_NAMES),
    }


@app.get("/ready")
def ready():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return {"ready": True}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    rows = [sample.as_row() for sample in body.samples]
    return PredictResponse(predictions=predictor.predict(rows))
