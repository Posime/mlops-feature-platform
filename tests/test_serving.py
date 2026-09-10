import pytest
from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

def test_healthz_endpoint():
    with TestClient(app) as test_client:
        response = test_client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["onnx_loaded"] is True
        assert data["redis_connected"] is True


def test_predict_endpoint_valid():
    with TestClient(app) as test_client:
        payload = {
            "user_id": 1015,
            "transaction_amount": 150.75
        }
        # Actual timed response
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["user_id"] == 1015
        assert 0.0 <= data["default_probability"] <= 1.0
        assert isinstance(data["is_default"], bool)
        assert data["latency_ms"] < 50.0  # Safe local CI threshold (< 50ms)
        assert "credit_score" in data["retrieved_features"]


def test_predict_validation_error():
    with TestClient(app) as test_client:
        # Invalid payload: negative transaction amount
        payload = {
            "user_id": 1015,
            "transaction_amount": -50.0
        }
        response = test_client.post("/v1/predict", json=payload)
        assert response.status_code == 422