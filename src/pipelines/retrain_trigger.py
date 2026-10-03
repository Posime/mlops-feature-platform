import mlflow
from pathlib import Path
import subprocess
import sys
from mlflow.tracking import MlflowClient

MODEL_NAME = "credit_risk_xgboost"


def run_drift_check() -> bool:
    """Executes the drift detector subprocess and checks exit status."""
    print("🔍 [ORCHESTRATOR] Running statistical drift inspection...")
    result = subprocess.run(
        [sys.executable, "src/monitoring/drift_detector.py"],
        capture_output=False,
    )
    # Drift detector returns 1 when drift is detected
    return result.returncode == 1


def trigger_dvc_pipeline():
    """Executes DVC reproduction to train a new model on updated data."""
    print("\n⚙️ [ORCHESTRATOR] Triggering automated DVC pipeline retraining...")
    cmd = ["uv", "run", "dvc", "repro", "-f"]
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError("DVC pipeline execution failed.")
    print("✅ [ORCHESTRATOR] Model retraining pipeline completed successfully.")


def evaluate_and_promote_challenger():
    """Compares the newly trained model against the current Champion."""
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()

    # Locate the active Champion model
    champion_versions = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champion_run = client.get_run(champion_versions.run_id)
    champion_auc = champion_run.data.metrics.get("roc_auc", 0.0)

    # Locate the most recent training run (the Challenger)
    experiment = client.get_experiment_by_name("Credit_default_training")
    if not experiment:
        raise RuntimeError("Experiment 'credit_default_training' not found in MLflow.")
    latest_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not latest_runs:
        raise RuntimeError("No recent MLflow training runs found.")

    challenger_run = latest_runs[0]
    challenger_auc = challenger_run.data.metrics.get("roc_auc", 0.0)

    print("\n⚖️ [MODEL GOVERNANCE] Evaluating Champion vs. Challenger:")
    print(
        f"  • Champion Model (v{champion_versions.version}) ROC-AUC:   {champion_auc:.4f}"
    )
    print(
        f"  • Challenger Model ({challenger_run.info.run_id[:8]}) ROC-AUC: {challenger_auc:.4f}"
    )

    # Register the newly trained Challenger
    model_uri = f"runs:/{challenger_run.info.run_id}/model"
    new_version = mlflow.register_model(model_uri, MODEL_NAME)

    # Promotion Gate: Challenger must strictly beat Champion
    if challenger_auc >= champion_auc:
        print(
            f"🎉 [PROMOTION] Challenger beats Champion. Promoting v{new_version.version} to @champion."
        )
        client.set_registered_model_alias(MODEL_NAME, "champion", new_version.version)
    else:
        print(
            f"⚠️ [REJECTED] Challenger (ROC-AUC {challenger_auc:.4f}) did not beat Champion (ROC-AUC {champion_auc:.4f})."
        )
        client.set_registered_model_alias(MODEL_NAME, "challenger", new_version.version)


def main():
    drift_detected = run_drift_check()

    if not drift_detected:
        print("✅ [ORCHESTRATOR] No drift detected. Serving stack remains stable.")
        sys.exit(0)

    print(
        "🚨 [ORCHESTRATOR] Statistical drift confirmed. Initiating self-healing workflow."
    )
    trigger_dvc_pipeline()
    evaluate_and_promote_challenger()
    print("🚀 [ORCHESTRATOR] Self-healing workflow successfully finished.")


if __name__ == "__main__":
    main()
