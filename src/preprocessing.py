"""Shared preprocessing utilities for training and inference."""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Binary features that map to 0/1
BINARY_MAPPINGS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

# Categorical features to one-hot encode
CATEGORICAL_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# Numeric features to scale
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary categorical features to 0/1."""
    df = df.copy()
    for col, mapping in BINARY_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def one_hot_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode remaining categorical features."""
    df = df.copy()
    cols_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    return df


def align_features(
    df: pd.DataFrame, expected_columns: List[str]
) -> pd.DataFrame:
    """Align dataframe columns with expected feature columns from training."""
    df = df.copy()
    # Add missing columns with 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    # Select only expected columns in correct order
    df = df[expected_columns]
    return df


def scale_numeric_features(
    df: pd.DataFrame, scaler: Optional[StandardScaler] = None, fit: bool = False
) -> pd.DataFrame:
    """Scale numeric features using StandardScaler."""
    df = df.copy()
    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]

    if not numeric_cols:
        return df, scaler

    if fit:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif scaler is not None:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    else:
        logger.warning("No scaler provided and fit=False. Numeric features not scaled.")

    return df, scaler


def preprocess_for_training(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Full preprocessing pipeline for training data.

    Returns:
        X: Preprocessed features
        y: Target variable
        scaler: Fitted StandardScaler
    """
    logger.info("Starting preprocessing for training")
    df = df.copy()

    # Drop ID column if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Handle TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Convert SeniorCitizen to string
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Encode target
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # Encode binary features
    X = encode_binary_features(X)

    # One-hot encode categoricals
    X = one_hot_encode_categoricals(X)

    # Scale numeric features
    X, scaler = scale_numeric_features(X, fit=True)

    logger.info(f"Preprocessing complete. Features shape: {X.shape}")
    return X, y, scaler


def preprocess_for_inference(
    data: Dict[str, Any],
    expected_columns: List[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Preprocess a single customer input for prediction.

    Args:
        data: Raw customer data dictionary
        expected_columns: Feature columns from training
        scaler: Fitted StandardScaler

    Returns:
        Preprocessed DataFrame ready for prediction
    """
    df = pd.DataFrame([data])

    # Handle TotalCharges if present
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Convert SeniorCitizen to string
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Encode binary features
    df = encode_binary_features(df)

    # One-hot encode categoricals
    df = one_hot_encode_categoricals(df)

    # Align with training columns
    df = align_features(df, expected_columns)

    # Scale numeric features
    df, _ = scale_numeric_features(df, scaler=scaler)

    return df


def preprocess_batch_csv(
    df: pd.DataFrame,
    expected_columns: List[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Preprocess a batch of customers from CSV.

    Args:
        df: Raw customer data DataFrame
        expected_columns: Feature columns from training
        scaler: Fitted StandardScaler

    Returns:
        Preprocessed DataFrame ready for prediction
    """
    df = df.copy()

    # Handle TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Convert SeniorCitizen to string
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Encode binary features
    df = encode_binary_features(df)

    # One-hot encode categoricals
    df = one_hot_encode_categoricals(df)

    # Align with training columns
    df = align_features(df, expected_columns)

    # Scale numeric features
    df, _ = scale_numeric_features(df, scaler=scaler)

    return df
