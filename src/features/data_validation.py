import pandera.pandas as pa
from pandera.typing import Series

# from datetime import datetime


class CreditFeatureMatrixSchema(pa.DataFrameModel):
    """Declarative schema contract for the processed feature matrix."""

    user_id: Series[int] = pa.Field(ge=1000, le=2000, description="Customer identification key")
    event_timestamp: Series[pa.DateTime] = pa.Field(description="Event timestamp")

    transaction_amount: Series[float] = pa.Field(
        ge=0.0, description="Transaction amount in USD must be non-negative"
    )
    account_balance: Series[float] = pa.Field(description="Account balance")
    credit_score: Series[int] = pa.Field(
        ge=300, le=850, description="FICO score bounded between 300 and 850"
    )
    failed_transactions_24h: Series[int] = pa.Field(
        ge=0, le=50, description="Count of recent failed transactions"
    )

    target_default: Series[int] = pa.Field(isin=[0, 1], description="Binary classification target")

    class Config:
        strict = True  # Reject any extra unexpected columns in the DataFrame
        coerce = True  # Attempt data type coercion before throwing validation errors
