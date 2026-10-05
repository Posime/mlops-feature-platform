import os
import sys
from datetime import datetime, timezone

# Ensure local host executions target localhost:6379 if unset
os.environ.setdefault("REDIS_CONNECTION_STRING", "localhost:6379")

from feast import FeatureStore  # noqa: E402


def run_materialization():
    store = FeatureStore(repo_path="src/features")
    print("⏳ Materializing offline features into Redis online store...")
    try:
        store.materialize(
            start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_date=datetime.now(timezone.utc),
        )
        print("✅ Online feature store materialized successfully.")
    except Exception as exc:
        print(f"❌ Materialization failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_materialization()