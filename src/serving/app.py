import time
import os
from contextlib import asynccontextmanager
import numpy as np
import onnxruntime as rt
from feast import FeatureStore
from fastapi import FastAPI, HTTPException, status
from src.serving.schemas import (
    CreditPredictionRequest,
    CreditPredictionResponse,
    HealthResponse
)

# Runtime global state
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize Feast Feature Store
    feast_repo_path = os.getenv("FEAST_REPO_PATH", "src/features")
    state["feature_store"] = FeatureStore(repo_path=feast_repo_path)

    # 2. Initialize ONNX Runtime Engine
    onnx_path = os.getenv("MODEL_PATH", "models/model.onnx")
    session_options = rt.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    state["onnx_session"] = rt.InferenceSession(
        onnx_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )
    state["input_name"] = state["onnx_session"].get_inputs()[0].name

    # 3. Kernel & Connection Warm-up
    print("🔥 [SERVING INIT] Warming up Redis connection pool and ONNX engine...")
    try:
        # Warm Redis pool
        _ = state["feature_store"].get_online_features(
            features=["user_credit_features:credit_score"],
            entity_rows=[{"user_id": 1000}]
        )
        # Warm ONNX runtime kernels
        dummy_tensor = np.zeros((1, 4), dtype=np.float32)
        _ = state["onnx_session"].run(None, {state["input_name"]: dummy_tensor})
        print("✅ [SERVING INIT] Warm-up complete. Steady-state latency achieved.")
    except Exception as e:
        print(f"⚠️ [SERVING INIT] Warm-up warning: {e}")

    yield

    state.clear()
    print("🛑 [SERVING SHUTDOWN] Serving resources cleaned up.")

app = FastAPI(
    title="Real-Time Credit Decisioning Engine",
    description="Low-latency inference pipeline combining Feast online Redis lookups with ONNX Runtime.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/healthz", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    store: FeatureStore = state.get("feature_store")
    session = state.get("onnx_session")

    redis_ok = False
    if store is not None:
        try:
            # Check Redis connection via active online feature query
            _ = store.get_online_features(
                features=["user_credit_features:credit_score"],
                entity_rows=[{"user_id": 1000}]
            )
            redis_ok = True
        except Exception:
            redis_ok = False

    return HealthResponse(
        status="healthy" if (redis_ok and session is not None) else "unhealthy",
        redis_connected=redis_ok,
        onnx_loaded=session is not None
    )


@app.post("/v1/predict", response_model=CreditPredictionResponse, status_code=status.HTTP_200_OK)
async def predict(request: CreditPredictionRequest):
    t_start = time.perf_counter()
    store: FeatureStore = state.get("feature_store")
    session: rt.InferenceSession = state.get("onnx_session")

    if not store or not session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serving engines not initialized."
        )

    # 1. Fetch Online Features from Redis (< 2ms)
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

    # 2. Extract and Impute Cold-Start Values
    account_balance = feast_response["account_balance"][0]
    credit_score = feast_response["credit_score"][0]
    failed_tx = feast_response["failed_transactions_24h"][0]

    # Fallback to neutral default values if user is absent from Redis cache
    account_balance = float(account_balance) if account_balance is not None else 0.0
    credit_score = float(credit_score) if credit_score is not None else 600.0
    failed_tx = float(failed_tx) if failed_tx is not None else 0.0

    retrieved_features = {
        "transaction_amount": request.transaction_amount,
        "account_balance": account_balance,
        "credit_score": credit_score,
        "failed_transactions_24h": failed_tx
    }

    # 3. Assemble ONNX Tensor: shape [1, 4]
    input_tensor = np.array([[
        request.transaction_amount,
        account_balance,
        credit_score,
        failed_tx
    ]], dtype=np.float32)

    # 4. Execute ONNX Runtime Inference (< 0.5ms)
    try:
        outputs = session.run(None, {state["input_name"]: input_tensor})
        # outputs[0] = labels, outputs[1] = probabilities [{0: p0, 1: p1}]
        probabilities = outputs[1]
        default_prob = float(probabilities[0][1]) if isinstance(probabilities, list) else float(probabilities[0, 1])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ONNX inference failure: {str(e)}"
        )

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    return CreditPredictionResponse(
        user_id=request.user_id,
        default_probability=round(default_prob, 4),
        is_default=default_prob >= 0.5,
        latency_ms=round(total_latency_ms, 2),
        retrieved_features=retrieved_features
    )