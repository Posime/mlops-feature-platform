"""
Production Load Testing & Latency Profiling Suite.
Simulates concurrent user load against the credit scoring inference API
to benchmark latency percentiles (p50, p95, p99) and validate Feast/ONNX throughput.
"""

import random

from locust import HttpUser, between, events, task


class CreditScoringUser(HttpUser):
    # Wait between 10ms and 50ms between requests to simulate rapid real-world traffic
    wait_time = between(0.01, 0.05)

    # Candidate entity pool matching entities seeded into Feast Redis
    USER_ID_POOL = list(range(1000, 1025))

    @task(10)
    def predict_credit_score(self):
        """
        Primary Task (Weight 10): Realistic credit scoring evaluation requests.
        Draws from diverse user IDs to test Feast online lookup concurrency.
        """
        user_id = random.choice(self.USER_ID_POOL)
        transaction_amount = round(random.uniform(15.0, 1500.0), 2)

        payload = {
            "user_id": user_id,
            "transaction_amount": transaction_amount,
        }

        with self.client.post(
            "/v1/predict",
            json=payload,
            name="/v1/predict",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate response schema invariants under stress
                if "default_probability" in data and "is_default" in data:
                    response.success()
                else:
                    response.failure(f"Malformed response payload: {data}")
            elif response.status_code == 503:
                response.failure("Service Unavailable - Engine uninitialized")
            else:
                response.failure(f"HTTP Error {response.status_code}: {response.text}")

    @task(1)
    def check_health(self):
        """Secondary Task (Weight 1): Baseline service health probe."""
        self.client.get("/healthz", name="/healthz")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n🚀 ================= LOAD TEST COMMENCING =================")
    print("Targeting: FastAPI + Feast Redis + ONNX Runtime")
    print("SLA Targets: p50 <= 5ms | p95 <= 10ms | p99 <= 25ms")
    print("==========================================================\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("\n🛑 ================= LOAD TEST COMPLETED =================")
    print("Check Prometheus /metrics and Grafana for live telemetry traces.")
    print("==========================================================\n")
