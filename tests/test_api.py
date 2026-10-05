"""API contract tests with a mocked predictor."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from iris_classifier import api


def test_health_without_model(monkeypatch):
    monkeypatch.setattr(api, "predictor", None)
    client = TestClient(api.app, raise_server_exceptions=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "model_not_loaded"}


def test_predict_requires_model(monkeypatch):
    monkeypatch.setattr(api, "predictor", None)
    client = TestClient(api.app)
    response = client.post(
        "/predict",
        json={
            "samples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        },
    )
    assert response.status_code == 503


def test_predict_happy_path(monkeypatch):
    mock = MagicMock()
    mock.predict.return_value = [
        {
            "class_index": 0,
            "class_name": "setosa",
            "probabilities": {"setosa": 0.9, "versicolor": 0.05, "virginica": 0.05},
            "feature_order": [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ],
        }
    ]
    monkeypatch.setattr(api, "predictor", mock)
    client = TestClient(api.app)
    response = client.post(
        "/predict",
        json={
            "samples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"][0]["class_name"] == "setosa"
