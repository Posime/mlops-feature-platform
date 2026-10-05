import json
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "credit_default_training"


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def run_training():
    params = load_params()
    train_params = params["train"]
    target_col = params["features"]["target_column"]

    # 1. Load Processed Feature Matrix
    data_path = "data/processed/train_features.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing training data at {data_path}. Run pipeline ingestion first."
        )

    df = pd.read_parquet(data_path)

    # 2. Separate Features and Target
    drop_cols = ["user_id", "event_timestamp", target_col]
    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols]
    y = df[target_col]

    # 3. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_params["test_size"],
        random_state=train_params["random_state"],
        stratify=y,
    )

    # 4. Connect to MLflow & Enable Autologging (with log_models=False for explicit saving)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.xgboost.autolog(log_models=False)

    with mlflow.start_run(run_name="xgboost_baseline_run") as run:
        run_id = run.info.run_id
        print(f"🚀 [MLFLOW RUN STARTED] ID: {run_id}")

        # 5. Train Model
        model = XGBClassifier(
            n_estimators=train_params["n_estimators"],
            learning_rate=train_params["learning_rate"],
            max_depth=train_params["max_depth"],
            subsample=train_params["subsample"],
            scale_pos_weight=train_params["scale_pos_weight"],
            random_state=train_params["random_state"],
            eval_metric="logloss",
        )

        model.fit(X_train, y_train)

        # 6. Compute Holdout Evaluation Metrics
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "val_roc_auc": float(roc_auc_score(y_test, y_prob)),
            "val_pr_auc": float(average_precision_score(y_test, y_prob)),
            "val_f1_score": float(f1_score(y_test, y_pred)),
            "val_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "val_recall": float(recall_score(y_test, y_pred)),
        }

        print("\n📈 [HOLDOUT EVALUATION METRICS]")
        for k, v in metrics.items():
            print(f" - {k}: {v:.4f}")

        # Log Custom Validation Metrics & Parameters
        mlflow.log_metrics(metrics)
        mlflow.log_params({"features_list": feature_cols})

        # 7. Generate & Save Diagnostic Plots
        os.makedirs("reports/figures", exist_ok=True)

        # ROC Curve
        fig, ax = plt.subplots(figsize=(6, 4))
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="XGBoost")
        plt.title("Validation ROC-AUC Curve")
        plt.tight_layout()
        roc_plot_path = "reports/figures/roc_curve.png"
        plt.savefig(roc_plot_path)
        plt.close()
        mlflow.log_artifact(roc_plot_path, artifact_path="plots")

        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
        plt.title("Validation Confusion Matrix")
        plt.tight_layout()
        cm_plot_path = "reports/figures/confusion_matrix.png"
        plt.savefig(cm_plot_path)
        plt.close()
        mlflow.log_artifact(cm_plot_path, artifact_path="plots")

        # 8. Log Model Artifact
        mlflow.xgboost.log_model(
            xgb_model=model, artifact_path="model", input_example=X_train.head(3)
        )

        # 9. Save Local Artifacts
        os.makedirs("reports", exist_ok=True)
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        os.makedirs("models", exist_ok=True)
        model.save_model("models/model.json")
        print(f"\n✅ [MODEL PERSISTED] Saved model -> models/model.json")


if __name__ == "__main__":
    run_training()
