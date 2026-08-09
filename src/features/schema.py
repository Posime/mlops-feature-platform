from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float32, Int64

# 1. Define Primary Entity (Join Key for Online Lookups)
user_entity = Entity(
    name="user_id",
    join_keys=["user_id"],
    value_type=ValueType.INT64,
    description="Customer unique identification key"
)

# 2. Declare Offline Source (Created in Day 1)
raw_credit_source = FileSource(
    name="raw_credit_source",
    path="../../data/raw/credit_transactions.parquet",
    timestamp_field="event_timestamp"
)

# 3. Declare Feature View
user_credit_fv = FeatureView(
    name="user_credit_features",
    entities=[user_entity],
    ttl=timedelta(days=30),  # Lookback window for historical joins
    schema=[
        Field(name="transaction_amount", dtype=Float32),
        Field(name="account_balance", dtype=Float32),
        Field(name="credit_score", dtype=Int64),
        Field(name="failed_transactions_24h", dtype=Int64),
    ],
    online=True,
    source=raw_credit_source,
    tags={"team": "risk_analytics"}
)