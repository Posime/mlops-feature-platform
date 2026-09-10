from pydantic import BaseModel, Field
from typing import Dict, Any


class CreditPredictionRequest(BaseModel):
    user_id: int = Field(..., ge=1000, le=9999, description="Unique account identifier")
    transaction_amount: float = Field(..., gt=0.0, le=50000.0, description="Incoming real-time transaction value")

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": 1015,
                "transaction_amount": 249.50
            }
        }
    }


class CreditPredictionResponse(BaseModel):
    user_id: int
    default_probability: float = Field(..., description="Model probability output for credit default risk")
    is_default: bool = Field(..., description="Binary thresholded decision (threshold=0.5)")
    latency_ms: float = Field(..., description="Internal pipeline execution time in milliseconds")
    retrieved_features: Dict[str, Any] = Field(..., description="Feature snapshot fetched from Feast Redis")


class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    onnx_loaded: bool