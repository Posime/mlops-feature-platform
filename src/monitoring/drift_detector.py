"""
Unified Statistical & Visual Drift Detection Pipeline.
Combines custom vectorized PSI and SciPy Kolmogorov-Smirnov statistical tests
with Evidently AI interactive HTML reporting and automated quality test suites.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# pyright: reportMissingImports=false
try:
    try:
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        from evidently.metric_preset.data_drift import DataDriftPreset

    from evidently import Report

    try:
        from evidently.test_preset import DataDriftTestPreset
    except ImportError:
        from evidently.test_preset.data_drift import DataDriftTestPreset

    from evidently.test_suite import TestSuite
except ImportError:
    try:
        from evidently.presets import DataDriftPreset
        from evidently.report import Report
        from evidently.test_suite import TestSuite

        try:
            from evidently.test_preset import DataDriftTestPreset
        except ImportError:
            DataDriftTestPreset = None
    except ImportError:
        DataDriftPreset = None
        Report = None
        DataDriftTestPreset = None
        TestSuite = None

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
BASELINE_PATH = Path("data/processed/train_features.parquet")
INFERENCE_LOGS_DIR = Path("data/inference_logs")
REPORTS_DIR = Path("reports/drift")

EVALUATION_FEATURES = [
    "transaction_amount",
    "account_balance",
    "credit_score",
    "failed_transactions_24h",
]


# -----------------------------------------------------------------------------
# Custom Mathematical Formulations (Lightweight & Deterministic)
# -----------------------------------------------------------------------------
def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_buckets: int = 10) -> float:
    """
    Calculate the Population Stability Index (PSI) between baseline and target distributions.
    Applies quantile bucket bounds with Laplace smoothing to prevent division by zero.
    """
    if len(baseline) == 0 or len(target) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(baseline, percentiles)
    bucket_bounds[0] = -np.inf
    bucket_bounds[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline, bins=bucket_bounds)
    target_counts, _ = np.histogram(target, bins=bucket_bounds)

    # Laplace smoothing
    p = np.where(baseline_counts == 0, 1e-4, baseline_counts) / len(baseline)
    q = np.where(target_counts == 0, 1e-4, target_counts) / len(target)

    psi_value = np.sum((p - q) * np.log(p / q))
    return float(psi_value)


# -----------------------------------------------------------------------------
# Data Loading & Ingestion
# -----------------------------------------------------------------------------
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the reference parquet dataset and parse incoming daily JSONL inference logs."""
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline training matrix missing at: {BASELINE_PATH}")

    baseline_df = pd.read_parquet(BASELINE_PATH)

    log_files = list(INFERENCE_LOGS_DIR.glob("inferences_*.jsonl"))
    if not log_files:
        print("⚠️ [DRIFT DETECTOR] No inference logs found for drift analysis.")
        return baseline_df, pd.DataFrame()

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
                    print(
                        f"⚠️ [DRIFT DETECTOR] Skipping malformed JSON line {line_no} in {file.name}"
                    )
                    continue

    target_df = pd.DataFrame(records)
    return baseline_df, target_df


# -----------------------------------------------------------------------------
# Evidently Visual & Suite Artifact Generation
# -----------------------------------------------------------------------------
def generate_evidently_artifacts(
    baseline_df: pd.DataFrame, target_df: pd.DataFrame, reports_dir: Path
) -> None:
    """Generate both the interactive HTML diagnostic dashboard and test artifacts."""
    ref_aligned = baseline_df[EVALUATION_FEATURES].copy()
    cur_aligned = target_df[EVALUATION_FEATURES].copy()

    html_output_path = reports_dir / "drift_report.html"

    try:
        from evidently.legacy.metric_preset import DataDriftPreset
        from evidently.legacy.report import Report

        print("📊 [EVIDENTLY] Generating visual HTML distribution dashboard...")
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_aligned, current_data=cur_aligned)
        report.save_html(str(html_output_path))
        print(f"   -> Visual HTML report saved to: {html_output_path}")

    except Exception as e:
        print(f"⚠️ [EVIDENTLY] Error generating visual dashboard: {e}")
        print("   Continuing with custom statistical report.")


# -----------------------------------------------------------------------------
# Main Analysis Pipeline
# -----------------------------------------------------------------------------
def run_drift_analysis(
    psi_threshold: float = 0.25,
    ks_alpha: float = 0.05,
    min_records: int = 20,
) -> Dict[str, Any]:
    """Execute dual-layer drift evaluation across baseline and live inference logs."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_df, target_df = load_datasets()

    if len(target_df) < min_records:
        print(
            f"⚠️ [DRIFT DETECTOR] Insufficient records ({len(target_df)} found, "
            f"minimum {min_records} needed) for statistically powered drift testing."
        )
        return {"status": "insufficient_data", "retrain_recommended": False}

    drift_report: Dict[str, Any] = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "baseline_sample_size": len(baseline_df),
        "target_sample_size": len(target_df),
        "features": {},
        "retrain_recommended": False,
    }

    print("\n🔍 ================== STATISTICAL DRIFT REPORT ==================")
    print(
        f"Evaluating {len(target_df)} live inferences against "
        f"{len(baseline_df)} baseline records\n"
    )

    # Statistical Evaluation (SciPy + Custom PSI)
    for col in EVALUATION_FEATURES:
        if col not in baseline_df.columns or col not in target_df.columns:
            print(f"⚠️ Column '{col}' not present across both datasets. Skipping.")
            continue

        base_vals = baseline_df[col].dropna().to_numpy()
        target_vals = target_df[col].dropna().to_numpy()

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

    # Save Custom Statistical JSON Report
    custom_json_path = REPORTS_DIR / "drift_report.json"
    with open(custom_json_path, "w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=2)
    print(f"📁 Statistical summary report saved to: {custom_json_path}")

    # Generate Evidently HTML & JSON Artifacts
    generate_evidently_artifacts(baseline_df, target_df, REPORTS_DIR)

    return drift_report


if __name__ == "__main__":
    report = run_drift_analysis()
    if report.get("retrain_recommended"):
        print(
            "\n🚨 ACTION REQUIRED: Statistically significant drift detected. "
            "Pipeline recommends automated retraining."
        )
        sys.exit(1)
    else:
        print("\n✅ Distributions within acceptable bounds. No retraining required.")
        sys.exit(0)
