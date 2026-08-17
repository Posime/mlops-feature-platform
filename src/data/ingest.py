import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_params():
    with open("params.yaml", 'r') as f:
        return yaml.safe_load(f)['ingest']

def generate_mock_credit_data():
    """Generates synthetic customer transaction features with point-in-time timestamps."""
    params = load_params()
    np.random.seed(params['random_seed'])
    now = datetime.now()

    num_records = params["num_records"]
    user_ids = np.random.randint(params["user_id_min"], params["user_id_max"], size=num_records)
    timestamps = [now - timedelta(hours=int(x)) for x in np.random.randint(0, 720, size=num_records)]
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "event_timestamp": timestamps,
        "transaction_amount": np.random.uniform(5.0, 1500.0, size=num_records),
        "account_balance": np.random.uniform(100.0, 50000.0, size=num_records),
        "credit_score": np.random.randint(300, 850, size=num_records),
        "failed_transactions_24h": np.random.randint(0, 5, size=num_records),
        "target_default": np.random.choice([0, 1], size=num_records, p=[0.9, 0.1])
    })
    
    # Sort chronologically
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/credit_transactions.parquet"
    df.to_parquet(output_path, index=False)
    print(f"✅ [DATA INGESTION] Generated {num_records} records -> {output_path}")

if __name__ == "__main__":
    generate_mock_credit_data()