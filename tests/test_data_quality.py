import os

import pandas as pd
import pytest

from src.features.data_validation import CreditFeatureMatrixSchema

PROCESSED_DATA_PATH = "data/processed/train_features.parquet"


@pytest.fixture
def processed_dataframe():
    """PyTest fixture to load the generated training feature matrix."""
    assert os.path.exists(PROCESSED_DATA_PATH), f"Data file missing at {PROCESSED_DATA_PATH}"
    return pd.read_parquet(PROCESSED_DATA_PATH)


def test_feature_matrix_schema(processed_dataframe):
    """Validates data types, ranges, and strict column constraints."""
    # This will raise pa.errors.SchemaError if data contracts are violated
    validated_df = CreditFeatureMatrixSchema.validate(processed_dataframe)
    assert validated_df is not None
    assert len(validated_df) > 0


def test_zero_null_values(processed_dataframe):
    """Guarantees feature store output contains zero null/missing values."""
    null_counts = processed_dataframe.isnull().sum().sum()
    assert null_counts == 0, f"Found {null_counts} null values in processed feature matrix!"


def test_target_class_distribution(processed_dataframe):
    """Ensures dataset contains both positive and negative target classes."""
    unique_targets = set(processed_dataframe["target_default"].unique())
    assert unique_targets == {0, 1}, f"Unexpected target labels: {unique_targets}"
