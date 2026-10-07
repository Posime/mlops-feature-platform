import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as rt
import prometheus_client
import redis.asyncio as aioredis
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, status
from feast import FeatureStore
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

from src.serving.schemas import (
    CreditPredictionRequest,
    CreditPredictionResponse,
    HealthResponse,
)

# =============================================================================
# 1. METRIC INSTRUMENTATION: PILLAR A - PREDICTION COUNTERS
# =============================================================================
PREDICTION_REQUESTS_TOTAL = Counter(
    name="credit_inference_requests_total",
    documentation="Cumulative count of credit scoring inference API requests",
    labelnames=["status", "is_default"],
)

# =============================================================================
# 1. METRIC INSTRUMENTATION: PILLAR B - LATENCY HISTOGRAMS
# =============================================================================
INFERENCE_LATENCY_SECONDS = Histogram(
    name="credit_inference_latency_seconds",
    documentation="Execution latency across inference stages in seconds",
    labelnames=["stage"],
    buckets=(
        0.001,  # 1.0ms
        0.0025,  # 2.5ms
        0.005,  # 5.0ms (Target p50)
        0.010,  # 10.0ms (Target p95)
        0.025,  # 25.0ms (Target p99)
        0.050,  # 50.0ms
        0.100,  # 100.0ms
        0.250,  # 250.0ms
        0.500,  # 500.0ms
    ),
)

# =============================================================================
# 1. METRIC INSTRUMENTATION: PILLAR C - FEATURE & PROBABILITY DISTRIBUTIONS
# =============================================================================
MODEL_OUTPUT_PROBABILITY_HISTOGRAM = Histogram(
    name="credit_prediction_probability_distribution",
    documentation="Distribution of credit default probabilities emitted by the ONNX model",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

FEATURE_TRANSACTION_AMOUNT_HISTOGRAM = Histogram(
    name="credit_feature_transaction_amount",
    documentation="Distribution of incoming transaction amounts requested",
    buckets=(10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0),
)

FEATURE_CREDIT_SCORE_HISTOGRAM = Histogram(
    name="credit_feature_retrieved_credit_score",
    documentation="Distribution of credit scores retrieved from Feast",
    buckets=(300.0, 450.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0),
)

# =============================================================================
# 1. METRIC INSTRUMENTATION: PILLAR D - SYSTEM HEALTH & DEPENDENCY GAUGES
# =============================================================================
STORE_CONNECTION_GAUGE = Gauge(
    name="credit_store_redis_connected",
    documentation="Health status of Feast Redis connection (1=Connected, 0=Disconnected)",
)

MODEL_LOADED_GAUGE = Gauge(
    name="credit_model_onnx_loaded",
    documentation="Health status of ONNX runtime session (1=Loaded, 0=Unloaded)",
)

# Number of requests between automated drift evaluations
DRIFT_EVALUATION_INTERVAL = 1000  # Set to 1000 for realistic production load testing

# In-memory runtime state shared across requests within a worker
state = {}


# =============================================================================
# 2. LIFESPAN MANAGEMENT (STARTUP & TEARDOWN)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set fallback connection string for local testing environments
    redis_conn_str = os.getenv("REDIS_CONNECTION_STRING", "localhost:6379")
    if ":" in redis_conn_str:
        redis_host, redis_port = redis_conn_str.split(":", 1)
    else:
        redis_host, redis_port = redis_conn_str, 6379

    # 1. Initialize Shared Async Redis Client for Cross-Worker Telemetry Counters
    try:
        state["redis_client"] = aioredis.Redis(
            host=redis_host,
            port=int(redis_port),
            decode_responses=True,
        )
        await state["redis_client"].ping()
        print("🔗 [SERVING INIT] Shared Redis client connected successfully.", flush=True)
    except Exception as exc:
        state["redis_client"] = None
        print(f"⚠️ [SERVING INIT] Redis connection warning: {exc}", flush=True)

    # 2. Initialize Feast Online Feature Store
    feast_repo_path = os.getenv("FEAST_REPO_PATH", "src/features")
    try:
        state["feature_store"] = FeatureStore(repo_path=feast_repo_path)
        STORE_CONNECTION_GAUGE.set(1.0)
    except Exception as exc:
        STORE_CONNECTION_GAUGE.set(0.0)
        print(f"❌ [FEAST INIT ERROR] Failed to connect to store: {exc}", flush=True)

    # 3. Initialize ONNX Runtime Session
    onnx_path = os.getenv("MODEL_PATH", "models/model.onnx")
    if not os.path.exists(onnx_path):
        MODEL_LOADED_GAUGE.set(0.0)
        raise RuntimeError(f"Missing ONNX model artifact at '{onnx_path}'.")

    session_options = rt.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    state["onnx_session"] = rt.InferenceSession(
        onnx_path, sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    state["input_name"] = state["onnx_session"].get_inputs()[0].name
    MODEL_LOADED_GAUGE.set(1.0)

    # 4. Model Warm-up Run to Prime Caches
    try:
        _ = state["feature_store"].get_online_features(
            features=["user_credit_features:credit_score"],
            entity_rows=[{"user_id": 1000}],
        )
        dummy_tensor = np.zeros((1, 4), dtype=np.float32)
        _ = state["onnx_session"].run(None, {state["input_name"]: dummy_tensor})
    except Exception as exc:
        print(f"⚠️ [SERVING INIT] Warm-up warning: {exc}", flush=True)

    yield

    # Clean up on shutdown
    if state.get("redis_client") is not None:
        await state["redis_client"].aclose()
    state.clear()
    STORE_CONNECTION_GAUGE.set(0.0)
    MODEL_LOADED_GAUGE.set(0.0)


# Telemetry logging setup
LOG_DIR = Path("data/inference_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_inference_event(payload: dict) -> None:
    """Appends prediction event to daily JSONL buffer for drift detection."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"inferences_{today}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def trigger_drift_detection_job() -> None:
    """
    Spawns the drift detection pipeline in a detached subprocess.
    Streams output unbuffered directly to stdout without blocking FastAPI worker loops.
    """
    try:
        print(
            "\n⚡ [MONITORING] Request milestone reached! Triggering drift detector subprocess...",
            flush=True,
        )
        result = subprocess.run(
            [sys.executable, "src/monitoring/drift_detector.py"],
            check=False,
        )
        print(
            f"✅ [MONITORING] Automated drift check completed"
            f"with exit code: {result.returncode}.\n",
            flush=True,
        )
    except Exception as exc:
        print(f"⚠️ [MONITORING] Drift check invocation error: {exc}", flush=True)


# =============================================================================
# 3. FASTAPI APPLICATION DEFINITION
# =============================================================================
app = FastAPI(
    title="Real-Time Credit Decisioning Engine",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/metrics")
async def metrics():
    """Prometheus telemetry scrape endpoint supporting multiprocess and standalone modes."""
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        data = generate_latest(prometheus_client.REGISTRY)

    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check():
    store = state.get("feature_store")
    session = state.get("onnx_session")
    ready = (store is not None) and (session is not None)

    return HealthResponse(
        status="healthy" if ready else "unhealthy",
        redis_connected=ready,
        onnx_loaded=session is not None,
    )


@app.post(
    "/v1/predict",
    response_model=CreditPredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def predict(request: CreditPredictionRequest, background_tasks: BackgroundTasks):
    t_start = time.perf_counter()
    store: FeatureStore = state.get("feature_store")
    session: rt.InferenceSession = state.get("onnx_session")

    if not store or not session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serving engines not initialized.",
        )

    # Record incoming request feature distribution
    FEATURE_TRANSACTION_AMOUNT_HISTOGRAM.observe(request.transaction_amount)

    # -----------------------------------------------------------------
    # 1. Feature Retrieval & Instrumentation
    # -----------------------------------------------------------------
    t_feast_start = time.perf_counter()
    try:
        feast_response = store.get_online_features(
            features=[
                "user_credit_features:account_balance",
                "user_credit_features:credit_score",
                "user_credit_features:failed_transactions_24h",
            ],
            entity_rows=[{"user_id": request.user_id}],
        ).to_dict()
        INFERENCE_LATENCY_SECONDS.labels(stage="feast_lookup").observe(
            time.perf_counter() - t_feast_start
        )
    except Exception as exc:
        PREDICTION_REQUESTS_TOTAL.labels(status="feast_error", is_default="none").inc()
        STORE_CONNECTION_GAUGE.set(0.0)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feast Redis retrieval failure: {str(exc)}",
        )

    # -----------------------------------------------------------------
    # 2. Extract & Observe Features
    # -----------------------------------------------------------------
    def get_feature(key: str, default: float) -> float:
        val = feast_response.get(f"user_credit_features:{key}") or feast_response.get(key)
        if val and len(val) > 0 and val[0] is not None:
            return float(val[0])
        return default

    account_balance = get_feature("account_balance", 0.0)
    credit_score = get_feature("credit_score", 600.0)
    failed_tx = get_feature("failed_transactions_24h", 0.0)

    # Observe retrieved feature distribution for drift monitoring
    FEATURE_CREDIT_SCORE_HISTOGRAM.observe(credit_score)

    retrieved_features = {
        "transaction_amount": request.transaction_amount,
        "account_balance": account_balance,
        "credit_score": credit_score,
        "failed_transactions_24h": failed_tx,
    }

    # -----------------------------------------------------------------
    # 3. Model Scoring & Latency Instrumentation
    # -----------------------------------------------------------------
    input_tensor = np.array(
        [[request.transaction_amount, account_balance, credit_score, failed_tx]],
        dtype=np.float32,
    )

    t_onnx_start = time.perf_counter()
    try:
        outputs = session.run(None, {state["input_name"]: input_tensor})
        probabilities = outputs[1]
        default_prob = (
            float(probabilities[0][1])
            if isinstance(probabilities, list)
            else float(probabilities[0, 1])
        )
        INFERENCE_LATENCY_SECONDS.labels(stage="onnx_inference").observe(
            time.perf_counter() - t_onnx_start
        )
    except Exception as exc:
        PREDICTION_REQUESTS_TOTAL.labels(status="onnx_error", is_default="none").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ONNX inference failure: {str(exc)}",
        )

    duration_sec = time.perf_counter() - t_start
    total_latency_ms = duration_sec * 1000.0
    is_default = bool(default_prob >= 0.5)

    # -----------------------------------------------------------------
    # 4. Record Decision, Probability & End-to-End Metrics
    # -----------------------------------------------------------------
    PREDICTION_REQUESTS_TOTAL.labels(
        status="success",
        is_default=str(is_default).lower(),
    ).inc()
    MODEL_OUTPUT_PROBABILITY_HISTOGRAM.observe(default_prob)
    INFERENCE_LATENCY_SECONDS.labels(stage="total").observe(duration_sec)

    # -----------------------------------------------------------------
    # 5. Drift Telemetry Logging (Non-blocking Background Task)
    # -----------------------------------------------------------------
    event_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": request.user_id,
        "transaction_amount": request.transaction_amount,
        "account_balance": account_balance,
        "credit_score": credit_score,
        "failed_transactions_24h": failed_tx,
        "default_probability": default_prob,
        "is_default": is_default,
    }
    background_tasks.add_task(log_inference_event, event_payload)

    # -----------------------------------------------------------------
    # 6. Automated Drift Detection Trigger (Shared Milestone Check)
    # -----------------------------------------------------------------
    try:
        redis_conn = state.get("redis_client")
        if redis_conn is not None:
            total_requests = await redis_conn.incr("metrics:total_inferences")
        else:
            counter_file = LOG_DIR / ".counter"
            with open(counter_file, "a+", encoding="utf-8") as f:
                f.seek(0)
                raw_val = f.read().strip()
                total_requests = int(raw_val) + 1 if raw_val.isdigit() else 1
                f.seek(0)
                f.truncate()
                f.write(str(total_requests))

        if total_requests % DRIFT_EVALUATION_INTERVAL == 0:
            print(
                f"🎯 [MONITORING] Milestone threshold ({DRIFT_EVALUATION_INTERVAL}) reached! "
                "Queueing drift detector background task...",
                flush=True,
            )
            background_tasks.add_task(trigger_drift_detection_job)
    except Exception as count_exc:
        print(f"⚠️ [MONITORING] Counter tracking warning: {count_exc}", flush=True)

    return CreditPredictionResponse(
        user_id=request.user_id,
        default_probability=round(default_prob, 4),
        is_default=is_default,
        latency_ms=round(total_latency_ms, 2),
        retrieved_features=retrieved_features,
    )
