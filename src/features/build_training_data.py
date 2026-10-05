import os

import pandas as pd
from feast import FeatureStore


def build_historical_dataset():
    """Generates an offline training dataset using point-in-time correct feature joins."""
    store = FeatureStore(repo_path="src/features")

    # 1. Load raw dataset to extract target labels and event observation timestamps
    raw_df = pd.read_parquet("data/raw/credit_transactions.parquet")

    # 2. Build Entity DataFrame (Join Keys + Timestamps + Ground Truth Target)
    entity_df = raw_df[["user_id", "event_timestamp", "target_default"]].copy()

    print(f"📊 [FEAST] Building point-in-time join for {len(entity_df)} observation events...")

    # 3. Request features as of the observation timestamps
    features_to_fetch = [
        "user_credit_features:transaction_amount",
        "user_credit_features:account_balance",
        "user_credit_features:credit_score",
        "user_credit_features:failed_transactions_24h",
    ]

    # point-in-line join logic
    training_data = store.get_historical_features(entity_df=entity_df, features=features_to_fetch)

    # 4. Convert to Pandas DataFrame
    training_df = training_data.to_df()

    # 5. Clean up missing/null values if features predated registration
    training_df = training_df.fillna(0)

    # 6. Save processed feature matrix for model training
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/train_features.parquet"
    training_df.to_parquet(output_path, index=False)

    print(f"✅ [TRAINING DATASET GENERATED] Saved {len(training_df)} rows to {output_path}")
    print("\nSample Training Feature Matrix:")
    print(training_df.head())


if __name__ == "__main__":
    build_historical_dataset()
