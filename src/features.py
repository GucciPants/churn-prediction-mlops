"""Data loading, cleaning, and feature engineering pipeline."""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Load raw CSV data."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data: handle missing values and type conversions."""
    logger.info("Cleaning data")
    df = df.copy()

    # Drop ID column if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric (contains spaces as missing values)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Handle missing values in TotalCharges (typically new customers with tenure=0)
    missing_total_charges = df["TotalCharges"].isna().sum()
    if missing_total_charges > 0:
        logger.info(f"Filling {missing_total_charges} missing TotalCharges with 0")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Convert SeniorCitizen to string to treat as categorical
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    logger.info("Data cleaning completed")
    return df


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary categorical features to 0/1."""
    df = df.copy()
    binary_mappings = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
    }
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target variable to 0/1."""
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    return df


def preprocess_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Preprocess features: encode categoricals and scale numerics."""
    logger.info("Preprocessing features")
    df = df.copy()

    # Encode target
    df = encode_target(df)
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Encode binary features
    X = encode_binary_features(X)

    # One-hot encode remaining categoricals
    categorical_to_encode = [c for c in CATEGORICAL_FEATURES if c in X.columns and c not in BINARY_FEATURES]
    if categorical_to_encode:
        X = pd.get_dummies(X, columns=categorical_to_encode, drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    if numeric_cols:
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    logger.info(f"Preprocessed features shape: {X.shape}")
    return X, y, scaler


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    logger.info(f"Splitting data with test_size={TEST_SIZE}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def build_feature_pipeline(raw_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Full pipeline: load, clean, preprocess, split."""
    df = load_data(raw_path)
    df = clean_data(df)
    X, y, scaler = preprocess_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, scaler
