import os

import matplotlib.pyplot as plt
import mlflow

MLFLOW_TRACKING_URI = "http://localhost:5000"


def test_tracking_server():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = "credit_default_tracking"
    mlflow.set_experiment(experiment_name)

    print(f"📡 [MLFLOW] Connected to Tracking Server at {MLFLOW_TRACKING_URI}")

    with mlflow.start_run(run_name="smoke_test_run") as run:
        run_id = run.info.run_id
        print(f"🚀 [RUN STARTED] Active Run ID: {run_id}")

        # 1. Log Hyperparameters
        mlflow.log_param("model_type", "smoke_test_classifier")
        mlflow.log_param("learning_rate", 0.05)

        # 2. Log Scalar Metrics
        mlflow.log_metric("val_roc_auc", 0.884)
        mlflow.log_metric("val_f1_score", 0.762)

        # 3. Generate and Log an Artifact Plot
        os.makedirs("mlflow_artifacts", exist_ok=True)
        sample_plot_path = "mlflow_artifacts/test_plot.png"

        plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3, 4], [0.5, 0.7, 0.82, 0.88], marker="o")
        plt.title("Sample Validation ROC-AUC Curve")
        plt.xlabel("Epoch")
        plt.ylabel("ROC-AUC")
        plt.tight_layout()
        plt.savefig(sample_plot_path)
        plt.close()

        mlflow.log_artifact(sample_plot_path, artifact_path="diagnostics")

        print("✅ [METRICS & ARTIFACTS LOGGED] Successfully persisted metadata to Postgres.")


if __name__ == "__main__":
    test_tracking_server()
