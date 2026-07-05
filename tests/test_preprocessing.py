"""Tests for shared preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    BINARY_MAPPINGS,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    align_features,
    encode_binary_features,
    one_hot_encode_categoricals,
    preprocess_batch_csv,
    preprocess_for_inference,
    preprocess_for_training,
    scale_numeric_features,
)


class TestBinaryEncoding:
    def test_encode_gender(self):
        df = pd.DataFrame({"gender": ["Male", "Female"]})
        result = encode_binary_features(df)
        assert result["gender"].tolist() == [1, 0]

    def test_encode_partner(self):
        df = pd.DataFrame({"Partner": ["Yes", "No"]})
        result = encode_binary_features(df)
        assert result["Partner"].tolist() == [1, 0]

    def test_encode_all_binary_features(self):
        df = pd.DataFrame({
            "gender": ["Male", "Female"],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "PhoneService": ["Yes", "No"],
            "PaperlessBilling": ["No", "Yes"],
        })
        result = encode_binary_features(df)
        assert result["gender"].tolist() == [1, 0]
        assert result["Partner"].tolist() == [1, 0]
        assert result["Dependents"].tolist() == [0, 1]
        assert result["PhoneService"].tolist() == [1, 0]
        assert result["PaperlessBilling"].tolist() == [0, 1]

    def test_missing_column_ignored(self):
        df = pd.DataFrame({"other_col": ["a", "b"]})
        result = encode_binary_features(df)
        assert "other_col" in result.columns
        # Should not add gender if not present
        assert "gender" not in result.columns


class TestOneHotEncoding:
    def test_encode_single_column(self):
        df = pd.DataFrame({"InternetService": ["DSL", "Fiber optic", "No"]})
        result = one_hot_encode_categoricals(df)
        assert "InternetService" not in result.columns
        assert "InternetService_Fiber optic" in result.columns
        assert "InternetService_No" in result.columns

    def test_encode_multiple_columns(self):
        df = pd.DataFrame({
            "InternetService": ["DSL", "Fiber optic"],
            "Contract": ["Month-to-month", "One year"],
        })
        result = one_hot_encode_categoricals(df)
        assert "InternetService" not in result.columns
        assert "Contract" not in result.columns
        # Should have one-hot encoded columns
        assert any("InternetService" in c for c in result.columns)
        assert any("Contract" in c for c in result.columns)

    def test_drop_first避免多重共线性(self):
        df = pd.DataFrame({"InternetService": ["DSL", "Fiber optic", "No"]})
        result = one_hot_encode_categoricals(df)
        # drop_first=True, so only 2 columns for 3 categories
        internet_cols = [c for c in result.columns if "InternetService" in c]
        assert len(internet_cols) == 2

    def test_empty_categorical_columns(self):
        df = pd.DataFrame({"tenure": [1, 2, 3]})
        result = one_hot_encode_categoricals(df)
        assert "tenure" in result.columns
        assert len(result.columns) == 1


class TestFeatureAlignment:
    def test_add_missing_columns(self):
        df = pd.DataFrame({"col1": [1, 2]})
        expected = ["col1", "col2", "col3"]
        result = align_features(df, expected)
        assert list(result.columns) == expected
        assert result["col2"].tolist() == [0, 0]
        assert result["col3"].tolist() == [0, 0]

    def test_reorder_columns(self):
        df = pd.DataFrame({"col3": [3], "col1": [1], "col2": [2]})
        expected = ["col1", "col2", "col3"]
        result = align_features(df, expected)
        assert list(result.columns) == expected

    def test_remove_extra_columns(self):
        df = pd.DataFrame({"col1": [1], "col2": [2], "col3": [3]})
        expected = ["col1", "col2"]
        result = align_features(df, expected)
        assert list(result.columns) == expected


class TestScaling:
    def test_fit_and_transform(self):
        df = pd.DataFrame({"tenure": [1, 2, 3], "MonthlyCharges": [10, 20, 30]})
        result, scaler = scale_numeric_features(df, fit=True)
        assert scaler is not None
        assert isinstance(scaler, StandardScaler)
        # Scaled values should have mean ~0
        assert abs(result["tenure"].mean()) < 0.01
        assert abs(result["MonthlyCharges"].mean()) < 0.01

    def test_transform_with_fitted_scaler(self):
        df_train = pd.DataFrame({"tenure": [1, 2, 3], "MonthlyCharges": [10, 20, 30]})
        _, scaler = scale_numeric_features(df_train, fit=True)

        df_test = pd.DataFrame({"tenure": [2], "MonthlyCharges": [20]})
        result, _ = scale_numeric_features(df_test, scaler=scaler)
        # Should be close to 0 (mean of training data)
        assert abs(result["tenure"].iloc[0]) < 0.1

    def test_no_scaler_warning(self):
        df = pd.DataFrame({"tenure": [1, 2, 3]})
        result, scaler = scale_numeric_features(df, fit=False)
        assert scaler is None
        # Values should be unchanged
        assert result["tenure"].tolist() == [1, 2, 3]


class TestPreprocessForTraining:
    def test_basic_preprocessing(self):
        df = pd.DataFrame({
            "customerID": ["C1", "C2"],
            "tenure": [1, 24],
            "MonthlyCharges": [29.85, 89.10],
            "TotalCharges": [29.85, 2138.70],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["Yes", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["One year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Mailed check", "Electronic check"],
            "Churn": ["No", "Yes"],
        })
        X, y, scaler = preprocess_for_training(df)
        # customerID should be dropped
        assert "customerID" not in X.columns
        # Churn should be encoded
        assert y.tolist() == [0, 1]
        # Scaler should be returned
        assert scaler is not None

    def test_handles_missing_total_charges(self):
        df = pd.DataFrame({
            "tenure": [0, 1],
            "TotalCharges": [" ", "29.85"],
            "MonthlyCharges": [29.85, 29.85],
            "gender": ["Male", "Male"],
            "Partner": ["No", "No"],
            "Dependents": ["No", "No"],
            "PhoneService": ["Yes", "Yes"],
            "Churn": ["No", "No"],
        })
        X, y, scaler = preprocess_for_training(df)
        assert X["TotalCharges"].isna().sum() == 0


class TestPreprocessForInference:
    def test_single_customer(self):
        expected_cols = ["tenure", "MonthlyCharges", "TotalCharges", "gender"]
        scaler = StandardScaler()
        scaler.fit(pd.DataFrame({"tenure": [1, 2], "MonthlyCharges": [10, 20], "TotalCharges": [10, 20]}))

        data = {
            "tenure": 12,
            "MonthlyCharges": 65.0,
            "TotalCharges": 780.0,
            "gender": "Male",
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
        }
        result = preprocess_for_inference(data, expected_cols, scaler)
        assert list(result.columns) == expected_cols
        assert len(result) == 1


class TestPreprocessBatchCSV:
    def test_batch_preprocessing(self):
        expected_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        scaler = StandardScaler()
        scaler.fit(pd.DataFrame({"tenure": [1, 2], "MonthlyCharges": [10, 20], "TotalCharges": [10, 20]}))

        df = pd.DataFrame({
            "tenure": [12, 24],
            "MonthlyCharges": [65.0, 89.0],
            "TotalCharges": [780.0, 2136.0],
        })
        result = preprocess_batch_csv(df, expected_cols, scaler)
        assert list(result.columns) == expected_cols
        assert len(result) == 2
