"""Tests for feature engineering pipeline."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features import clean_data, encode_binary_features, encode_target, load_data


class TestLoadData:
    def test_load_existing_file(self):
        # Create a temporary CSV file for testing
        test_data = pd.DataFrame({
            "customerID": ["1", "2"],
            "tenure": [1, 2],
            "Churn": ["Yes", "No"]
        })
        test_path = Path("data/raw/test_temp.csv")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_data.to_csv(test_path, index=False)
        
        df = load_data(test_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        
        # Cleanup
        test_path.unlink()

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_data(Path("data/raw/nonexistent_test.csv"))


class TestCleanData:
    def test_drop_customer_id(self):
        df = pd.DataFrame({"customerID": ["1"], "tenure": [1]})
        cleaned = clean_data(df)
        assert "customerID" not in cleaned.columns

    def test_convert_total_charges(self):
        df = pd.DataFrame({"TotalCharges": ["29.85", " ", "100.00"], "tenure": [1, 2, 3]})
        cleaned = clean_data(df)
        assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
        assert cleaned["TotalCharges"].isna().sum() == 0


class TestEncodeBinaryFeatures:
    def test_gender_encoding(self):
        df = pd.DataFrame({"gender": ["Male", "Female"]})
        encoded = encode_binary_features(df)
        assert encoded["gender"].tolist() == [1, 0]


class TestEncodeTarget:
    def test_churn_encoding(self):
        df = pd.DataFrame({"Churn": ["Yes", "No"]})
        encoded = encode_target(df)
        assert encoded["Churn"].tolist() == [1, 0]
