"""Train churn prediction models with MLflow tracking."""

import logging
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MLFLOW_TRACKING_URI,
    MODELS,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
)
from src.explainer import create_explainer, save_explainer
from src.features import build_feature_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Train a single model and log metrics with MLflow."""
    logger.info(f"Training {model_name}")

    if model_name == "random_forest":
        model = RandomForestClassifier(**MODELS["random_forest"])
    elif model_name == "xgboost":
        model = XGBClassifier(**MODELS["xgboost"])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    logger.info(
        f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}"
    )

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # MLflow logging (optional - skip if MLflow is not available)
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("churn_prediction")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(MODELS[model_name])
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("precision", report["1"]["precision"])
            mlflow.log_metric("recall", report["1"]["recall"])
            mlflow.sklearn.log_model(model, artifact_path="model")
            logger.info(f"MLflow run logged for {model_name}")
    except Exception as e:
        logger.warning(f"MLflow logging failed for {model_name}: {e}")
        logger.info("Model was still trained and saved locally")

    return model, {"accuracy": accuracy, "f1_score": f1, "roc_auc": roc_auc}


def save_model(model, model_name: str, models_dir: Path = MODELS_DIR):
    """Save trained model to disk."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")


def main():
    """Run full training pipeline."""
    logger.info("Starting training pipeline")

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = build_feature_pipeline(RAW_DATA_PATH)

    # Save scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")

    # Train models
    results = {}
    explainers = {}
    for model_name in MODELS.keys():
        model, metrics = train_model(model_name, X_train, y_train, X_test, y_test)
        save_model(model, model_name)
        results[model_name] = metrics

        # Create and save SHAP explainer
        explainer = create_explainer(model, model_name)
        explainers[model_name] = explainer
        explainer_path = MODELS_DIR / f"{model_name}_explainer.pkl"
        save_explainer(explainer, explainer_path)

    # Print summary
    logger.info("Training completed. Results:")
    for model_name, metrics in results.items():
        logger.info(f"  {model_name}: {metrics}")

    return results


if __name__ == "__main__":
    main()
