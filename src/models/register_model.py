import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "credit_default_training"
REGISTERED_MODEL_NAME = "credit_default_classifier"
MIN_ROC_AUC_THRESHOLD = 0.50  # Promotion quality gate


def register_champion_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 1. Fetch the experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

    # 2. Query the best run ordered by validation ROC-AUC
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_roc_auc DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No completed runs found in experiment.")

    best_run = runs[0]
    run_id = best_run.info.run_id
    val_roc_auc = best_run.data.metrics.get("val_roc_auc", 0.0)

    print(f"🔍 [TOP CANDIDATE] Run ID: {run_id} with val_roc_auc: {val_roc_auc:.4f}")

    # 3. Quality Gate Check
    if val_roc_auc < MIN_ROC_AUC_THRESHOLD:
        print(
            f"❌ [QUALITY GATE FAILED] val_roc_auc ({val_roc_auc:.4f}) "
            f"below threshold ({MIN_ROC_AUC_THRESHOLD})"
        )
        return

    # 4. Register Model in the MLflow Catalog
    model_uri = f"runs:/{run_id}/model"
    print(f"📦 [REGISTERING] Registering model from {model_uri} -> '{REGISTERED_MODEL_NAME}'...")

    model_version = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    # 5. Set Version Metadata & Description
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        description=(
            f"Candidate model promoted from Run {run_id}. Holdout ROC-AUC: {val_roc_auc:.4f}"
        ),
    )

    # 6. Assign Production Alias (@champion)
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME, alias="champion", version=model_version.version
    )

    print(
        f"✅ [PROMOTION COMPLETE] Model Version {model_version.version} assigned alias '@champion'"
    )


if __name__ == "__main__":
    register_champion_model()
