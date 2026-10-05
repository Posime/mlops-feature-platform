import random

from locust import HttpUser, between, task


class CreditInferenceUser(HttpUser):
    # Simulated think-time between consecutive user actions (100ms to 500ms)
    wait_time = between(0.1, 0.5)

    @task(9)
    def predict_credit_risk(self):
        payload = {
            "user_id": random.randint(1000, 1050),
            "transaction_amount": round(random.uniform(10.0, 5000.0), 2),
        }
        self.client.post("/v1/predict", json=payload, name="/v1/predict")

    @task(1)
    def health_check(self):
        """Simulate orchestrator (Kubernetes/ECS) polling healthz endpoint."""
        self.client.get("/healthz", name="/healthz")
