import time

import numpy as np
import onnxruntime as rt
import pandas as pd
from xgboost import XGBClassifier


def run_benchmark(num_iterations=1000):
    sample_input = np.array([[150.0, 3200.0, 680.0, 1.0]], dtype=np.float32)
    feature_names = [
        "transaction_amount",
        "account_balance",
        "credit_score",
        "failed_transactions_24h",
    ]
    sample_df = pd.DataFrame(sample_input, columns=feature_names)

    # 1. Native XGBoost Benchmark
    xgb_model = XGBClassifier()
    xgb_model.load_model("models/model.json")

    xgb_latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = xgb_model.predict_proba(sample_df)
        xgb_latencies.append((time.perf_counter() - t0) * 1000)

    # 2. ONNX Runtime Benchmark
    session = rt.InferenceSession("models/model.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    onnx_latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: sample_input})
        onnx_latencies.append((time.perf_counter() - t0) * 1000)

    print("\n⚡ ================= INFERENCE LATENCY BENCHMARK ================")
    print(f"🔥 Iterations: {num_iterations} single-record requests")
    print("----------------------------------------------------------------")
    print(
        f"Native XGBoost (Python) -> p50: {np.percentile(xgb_latencies, 50):.3f}ms | p95: {np.percentile(xgb_latencies, 95):.3f}ms | p99: {np.percentile(xgb_latencies, 99):.3f}ms"
    )
    print(
        f"ONNX Runtime (C++ Engine)-> p50: {np.percentile(onnx_latencies, 50):.3f}ms | p95: {np.percentile(onnx_latencies, 95):.3f}ms | p99: {np.percentile(onnx_latencies, 99):.3f}ms"
    )
    print("================================================================")


if __name__ == "__main__":
    run_benchmark()
