"""Data loading and feature engineering pipeline."""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.preprocessing import preprocess_for_training

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Load raw CSV data."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


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


def build_feature_pipeline(
    raw_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, "StandardScaler"]:
    """Full pipeline: load, preprocess, split.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    df = load_data(raw_path)
    X, y, scaler = preprocess_for_training(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, scaler
