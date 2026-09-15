import time
import os
from contextlib import asynccontextmanager
import numpy as np
import onnxruntime as rt
from feast import FeatureStore
from fastapi import FastAPI, HTTPException, status, Response
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, multiprocess, CONTENT_TYPE_LATEST

from src.serving.schemas import (
    CreditPredictionRequest,
    CreditPredictionResponse,
    HealthResponse
)

# -------------------------------------------------------------------
# Custom Prometheus Metrics
# -------------------------------------------------------------------
PREDICTION_COUNTER = Counter(
    "model_prediction_total",
    "Total number of model predictions served",
    ["decision"]
)

PREDICTION_LATENCY_HISTOGRAM = Histogram(
    "model_prediction_latency_seconds",
    "Time spent running feature retrieval and model inference",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PREDICTION_PROBABILITY_HISTOGRAM = Histogram(
    "model_prediction_probability",
    "Distribution of output default probabilities (for drift detection)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    feast_repo_path = os.getenv("FEAST_REPO_PATH", "src/features")
    state["feature_store"] = FeatureStore(repo_path=feast_repo_path)

    onnx_path = os.getenv("MODEL_PATH", "models/model.onnx")
    if not os.path.exists(onnx_path):
        raise RuntimeError(f"Missing ONNX model artifact at '{onnx_path}'.")

    session_options = rt.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    state["onnx_session"] = rt.InferenceSession(
        onnx_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )
    state["input_name"] = state["onnx_session"].get_inputs()[0].name

    try:
        _ = state["feature_store"].get_online_features(
            features=["user_credit_features:credit_score"],
            entity_rows=[{"user_id": 1000}]
        )
        dummy_tensor = np.zeros((1, 4), dtype=np.float32)
        _ = state["onnx_session"].run(None, {state["input_name"]: dummy_tensor})
    except Exception as e:
        print(f"⚠️ [SERVING INIT] Warm-up warning: {e}")

    yield
    state.clear()


app = FastAPI(
    title="Real-Time Credit Decisioning Engine",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/metrics")
def metrics():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)



@app.get("/healthz", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check():
    store = state.get("feature_store")
    session = state.get("onnx_session")
    ready = (store is not None) and (session is not None)

    return HealthResponse(
        status="healthy" if ready else "unhealthy",
        redis_connected=ready,
        onnx_loaded=session is not None
    )


@app.post("/v1/predict", response_model=CreditPredictionResponse, status_code=status.HTTP_200_OK)
def predict(request: CreditPredictionRequest):
    t_start = time.perf_counter()
    store: FeatureStore = state.get("feature_store")
    session: rt.InferenceSession = state.get("onnx_session")

    if not store or not session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serving engines not initialized."
        )

    # 1. Fetch Online Features from Redis
    try:
        feast_response = store.get_online_features(
            features=[
                "user_credit_features:account_balance",
                "user_credit_features:credit_score",
                "user_credit_features:failed_transactions_24h"
            ],
            entity_rows=[{"user_id": request.user_id}]
        ).to_dict()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feast Redis retrieval failure: {str(e)}"
        )

    # 2. Extract & Impute Defaults
    account_balance = feast_response["account_balance"][0]
    credit_score = feast_response["credit_score"][0]
    failed_tx = feast_response["failed_transactions_24h"][0]

    account_balance = float(account_balance) if account_balance is not None else 0.0
    credit_score = float(credit_score) if credit_score is not None else 600.0
    failed_tx = float(failed_tx) if failed_tx is not None else 0.0

    retrieved_features = {
        "transaction_amount": request.transaction_amount,
        "account_balance": account_balance,
        "credit_score": credit_score,
        "failed_transactions_24h": failed_tx
    }

    # 3. ONNX Inference
    input_tensor = np.array([[
        request.transaction_amount,
        account_balance,
        credit_score,
        failed_tx
    ]], dtype=np.float32)

    try:
        outputs = session.run(None, {state["input_name"]: input_tensor})
        probabilities = outputs[1]
        default_prob = float(probabilities[0][1]) if isinstance(probabilities, list) else float(probabilities[0, 1])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ONNX inference failure: {str(e)}"
        )

    duration_sec = time.perf_counter() - t_start
    total_latency_ms = duration_sec * 1000.0
    is_default = default_prob >= 0.5

    # 4. Record Custom Telemetry
    PREDICTION_COUNTER.labels(decision="default" if is_default else "non_default").inc()
    PREDICTION_LATENCY_HISTOGRAM.observe(duration_sec)
    PREDICTION_PROBABILITY_HISTOGRAM.observe(default_prob)

    return CreditPredictionResponse(
        user_id=request.user_id,
        default_probability=round(default_prob, 4),
        is_default=is_default,
        latency_ms=round(total_latency_ms, 2),
        retrieved_features=retrieved_features
    )