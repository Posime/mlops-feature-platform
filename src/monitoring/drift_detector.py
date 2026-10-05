import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates the Population Stability Index (PSI) between two distributions."""
    if len(baseline) == 0 or len(target) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(baseline, percentiles)
    bucket_bounds[0] = -np.inf
    bucket_bounds[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline, bins=bucket_bounds)
    target_counts, _ = np.histogram(target, bins=bucket_bounds)

    # Laplace smoothing to prevent division by zero
    p = np.where(baseline_counts == 0, 1e-4, baseline_counts) / len(baseline)
    q = np.where(target_counts == 0, 1e-4, target_counts) / len(target)

    psi_value = np.sum((p - q) * np.log(p / q))
    return float(psi_value)


def run_drift_analysis(
    baseline_parquet_path: str = "data/processed/train_features.parquet",
    inference_logs_dir: str = "data/inference_logs",
    psi_threshold: float = 0.25,
    ks_alpha: float = 0.05,
) -> dict:
    """Evaluates covariate and prediction drift against the baseline training dataset."""  # noqa: E501
    baseline_path = Path(baseline_parquet_path)
    logs_dir = Path(inference_logs_dir)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline training matrix missing at: {baseline_path}")

    baseline_df = pd.read_parquet(baseline_path)

    log_files = list(logs_dir.glob("inferences_*.jsonl"))
    if not log_files:
        print("⚠️ No inference logs found for drift analysis.")
        return {"status": "insufficient_data"}

    records = []
    for file in log_files:
        with open(file, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                clean_line = raw_line.strip()
                if not clean_line:
                    continue
                try:
                    records.append(json.loads(clean_line))
                except json.JSONDecodeError:
                    # Gracefully bypass partially written or malformed records
                    print(f"⚠️ Skipping malformed JSON line {line_no} in {file.name}")
                    continue

    if len(records) < 20:
        print(
            f"⚠️ Insufficient records ({len(records)} found, minimum 20 needed) for statistical power."  # noqa: E501
        )
        return {"status": "insufficient_data"}

    target_df = pd.DataFrame(records)

    features = [
        "transaction_amount",
        "account_balance",
        "credit_score",
        "failed_transactions_24h",
    ]

    drift_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "sample_size": len(target_df),
        "features": {},
        "retrain_recommended": False,
    }

    print("\n🔍 ================== STATISTICAL DRIFT REPORT ==================")
    print(
        f"Analyzing {len(target_df)} live inferences against {len(baseline_df)} baseline records\n"
    )

    for col in features:
        base_vals = baseline_df[col].dropna().values
        target_vals = target_df[col].dropna().values

        ks_stat, ks_p_val = ks_2samp(base_vals, target_vals)
        psi_val = calculate_psi(base_vals, target_vals)

        is_drifted = (psi_val >= psi_threshold) or (ks_p_val < ks_alpha)

        drift_report["features"][col] = {
            "psi": round(psi_val, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_p_val), 4),
            "drift_detected": bool(is_drifted),
        }

        flag = "🚨 DRIFT" if is_drifted else "✅ STABLE"
        print(f"[{flag}] {col:<26} | PSI: {psi_val:.4f} | KS p-val: {ks_p_val:.4e}")

        if is_drifted:
            drift_report["retrain_recommended"] = True

    print("=================================================================\n")

    report_path = Path("reports/drift_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=2)

    return drift_report


if __name__ == "__main__":
    report = run_drift_analysis()
    if report.get("retrain_recommended"):
        print(
            "🚨 ACTION REQUIRED: Critical drift detected. Triggering automated retraining flow."  # noqa: E501
        )
        sys.exit(1)
    else:
        print("✅ Distributions within acceptable bounds. No retraining required.")
        sys.exit(0)
