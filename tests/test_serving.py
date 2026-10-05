from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture(scope="module")
def client():
    """Provides a managed TestClient instance that properly executes

    the FastAPI lifespan event (ONNX model loading & Redis connection).
    """
    with TestClient(app) as test_client:
        yield test_client


# =====================================================================
# 1. Integration Tests (Live Redis & Live ONNX)
# =====================================================================
def test_healthz_endpoint(client: TestClient):
    """Verifies that ONNX and Redis dependencies are actively connected."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["onnx_loaded"] is True
    assert data["redis_connected"] is True


def test_predict_endpoint_valid(client: TestClient):
    """Verifies end-to-end inference output contract and score boundaries."""
    payload = {
        "user_id": 1015,
        "transaction_amount": 150.75,
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200, f"Server returned 500 : {response.json()}"
    data = response.json()

    assert data["user_id"] == 1015
    assert 0.0 <= data["default_probability"] <= 1.0
    assert isinstance(data["is_default"], bool)
    # Relaxed latency ceiling to account for shared CI virtualized runner jitter
    assert data["latency_ms"] < 250.0
    assert "credit_score" in data["retrieved_features"]


def test_predict_validation_error(client: TestClient):
    """Verifies that schema constraint violations return HTTP 422 Unprocessable Entity."""
    payload = {
        "user_id": 1015,
        "transaction_amount": -50.0,  # Negative transaction violates Field(gt=0)
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422


# =====================================================================
# 2. Unit Tests (Mocked Dependencies)
# =====================================================================


@patch("src.serving.app.state")
def test_predict_successful_mock(mock_state, client: TestClient):
    """Verifies inference parsing using isolated mocked Feast and ONNX runtimes."""
    # Mock Feature Store
    mock_fs = MagicMock()
    mock_fs.get_online_features.return_value.to_dict.return_value = {
        "user_id": [1001],
        "account_balance": [12500.0],
        "credit_features:account_balance": [12500.0],
        "credit_score": [710.0],
        "credit_features:credit_score": [710.0],
        "failed_transactions_24h": [0.0],
        "credit_features:failed_transactions_24h": [0.0],
    }

    # Mock ONNX Session
    mock_session = MagicMock()
    mock_session.run.return_value = [
        np.array([0]),
        [{0: 0.85, 1: 0.15}],
    ]

    mock_dict = {
        "feature_store": mock_fs,
        "onnx_session": mock_session,
        "input_name": "float_input",
    }

    mock_state.get.side_effect = lambda k, default=None: mock_dict.get(k, default)

    mock_state.__getitem__.side_effect = lambda k: mock_dict[k]

    payload = {
        "user_id": 1001,
        "transaction_amount": 120.50,
    }

    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["user_id"] == 1001
    assert "default_probability" in res_data
    assert "is_default" in res_data
    assert "latency_ms" in res_data
    assert res_data["retrieved_features"]["account_balance"] == 12500.0
