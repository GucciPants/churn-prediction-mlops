"""SHAP model explanations for churn prediction."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def create_explainer(model, model_name: str = "random_forest"):
    """Create a SHAP TreeExplainer for the given model."""
    logger.info(f"Creating SHAP explainer for {model_name}")
    explainer = shap.TreeExplainer(model)
    return explainer


def get_feature_importance(explainer, model, X_sample: pd.DataFrame):
    """Get global feature importance from SHAP values."""
    logger.info("Computing global SHAP feature importance")
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values is a list [class_0, class_1]
    # We use class_1 (churn) importance
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    importance = np.abs(shap_values).mean(axis=0)
    feature_names = X_sample.columns.tolist()

    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)


def explain_prediction(explainer, X_input: pd.DataFrame, feature_names: list = None):
    """Explain a single prediction with SHAP values.

    Returns:
        dict with:
        - base_value: the expected value (baseline)
        - shap_values: list of {feature, value, contribution} dicts
        - top_positive: top features pushing toward churn
        - top_negative: top features pushing against churn
    """
    logger.info("Explaining prediction with SHAP")
    shap_values = explainer.shap_values(X_input)

    # For binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Get base value (expected value)
    base_value = float(explainer.expected_value)
    if isinstance(base_value, (list, np.ndarray)):
        base_value = (
            float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
        )

    # Build feature contributions
    contributions = []
    for i, feature in enumerate(X_input.columns):
        contributions.append(
            {
                "feature": feature,
                "value": float(X_input.iloc[0, i]),
                "contribution": float(shap_values[0, i]),
            }
        )

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    # Split into positive (pushing churn) and negative (pushing retain)
    top_positive = [c for c in contributions if c["contribution"] > 0][:5]
    top_negative = [c for c in contributions if c["contribution"] < 0][:5]

    return {
        "base_value": base_value,
        "shap_values": contributions[:10],
        "top_positive": top_positive,
        "top_negative": top_negative,
    }


def save_explainer(explainer, path: Path):
    """Save SHAP explainer to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(explainer, f)
    logger.info(f"SHAP explainer saved to {path}")


def load_explainer(path: Path):
    """Load SHAP explainer from disk."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        explainer = pickle.load(f)
    logger.info(f"SHAP explainer loaded from {path}")
    return explainer
