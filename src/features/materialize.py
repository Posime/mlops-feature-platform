from datetime import datetime, timedelta
from feast import FeatureStore

def run_materialization():
    store = FeatureStore(repo_path="src/features")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"🔄 [FEAST] Materializing features to Redis from {start_date} to {end_date}...")
    store.materialize(start_date, end_date)
    print("✅ [FEAST] Materialization to Redis complete.")

if __name__ == "__main__":
    run_materialization()