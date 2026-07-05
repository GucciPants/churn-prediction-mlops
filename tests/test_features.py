"""Tests for feature engineering pipeline."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features import build_feature_pipeline, load_data, split_data


class TestLoadData:
    def test_load_existing_file(self):
        # Create a temporary CSV file for testing
        test_data = pd.DataFrame(
            {"customerID": ["1", "2"], "tenure": [1, 2], "Churn": ["Yes", "No"]}
        )
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

    def test_loaded_dataframe_has_expected_columns(self):
        test_data = pd.DataFrame({
            "customerID": ["1"],
            "tenure": [1],
            "MonthlyCharges": [29.85],
            "Churn": ["No"],
        })
        test_path = Path("data/raw/test_cols.csv")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_data.to_csv(test_path, index=False)

        df = load_data(test_path)
        assert list(df.columns) == ["customerID", "tenure", "MonthlyCharges", "Churn"]

        test_path.unlink()


class TestSplitData:
    def test_split_ratio(self):
        X = pd.DataFrame({"col": range(100)})
        y = pd.Series([0] * 50 + [1] * 50)

        X_train, X_test, y_train, y_test = split_data(X, y)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_stratified_split(self):
        X = pd.DataFrame({"col": range(100)})
        y = pd.Series([0] * 80 + [1] * 20)  # Imbalanced

        X_train, X_test, y_train, y_test = split_data(X, y)

        # Check class distribution is maintained
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05  # Should be similar

    def test_reproducible_split(self):
        X = pd.DataFrame({"col": range(100)})
        y = pd.Series([0] * 50 + [1] * 50)

        split1 = split_data(X, y)
        split2 = split_data(X, y)

        # Same random state should produce same split
        assert split1[0].index.tolist() == split2[0].index.tolist()


class TestBuildFeaturePipeline:
    def test_full_pipeline_with_sample_data(self):
        # Create sample data
        test_data = pd.DataFrame({
            "customerID": ["C1", "C2", "C3", "C4"],
            "tenure": [1, 12, 24, 48],
            "MonthlyCharges": [29.85, 50.00, 70.00, 100.00],
            "TotalCharges": [29.85, 600.00, 1680.00, 4800.00],
            "gender": ["Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 0, 1, 1],
            "Partner": ["No", "Yes", "Yes", "No"],
            "Dependents": ["No", "Yes", "No", "Yes"],
            "PhoneService": ["Yes", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No", "Yes", "No", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["Yes", "No", "Yes", "No"],
            "OnlineBackup": ["No", "Yes", "No", "No"],
            "DeviceProtection": ["No", "No", "Yes", "No"],
            "TechSupport": ["Yes", "No", "Yes", "No"],
            "StreamingTV": ["No", "No", "No", "No"],
            "StreamingMovies": ["No", "No", "No", "No"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "Churn": ["Yes", "No", "No", "Yes"],
        })

        test_path = Path("data/raw/test_pipeline.csv")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_data.to_csv(test_path, index=False)

        X_train, X_test, y_train, y_test, scaler = build_feature_pipeline(test_path)

        # Check shapes
        assert X_train.shape[0] == 3  # 80% of 4
        assert X_test.shape[0] == 1  # 20% of 4

        # Check target is encoded
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

        # Check scaler is fitted
        assert scaler is not None
        assert hasattr(scaler, "mean_")

        # Cleanup
        test_path.unlink()
