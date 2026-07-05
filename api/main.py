"""FastAPI backend for churn prediction."""

import io
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.explainer import explain_prediction, load_explainer
from src.preprocessing import preprocess_batch_csv, preprocess_for_inference

# Load model and scaler
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Global artifacts
model = None
model_xgb = None
scaler = None
explainer = None
expected_columns = None


def load_artifacts():
    """Load trained models, scaler, and SHAP explainer.

    Raises:
        FileNotFoundError: If required model files are not found.
    """
    global model, model_xgb, scaler, explainer, expected_columns

    model_path = MODELS_DIR / "random_forest.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    explainer_path = MODELS_DIR / "random_forest_explainer.pkl"

    # Check required files exist
    missing = []
    if not model_path.exists():
        missing.append(str(model_path))
    if not scaler_path.exists():
        missing.append(str(scaler_path))

    if missing:
        raise FileNotFoundError(
            f"Required model files not found. Please run training first: {missing}"
        )

    # Load Random Forest
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded Random Forest model from {model_path}")

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info(f"Loaded scaler from {scaler_path}")

    # Store expected columns from model
    expected_columns = list(model.feature_names_in_)

    # Load SHAP explainer (optional)
    explainer = load_explainer(explainer_path)
    if explainer is None:
        logger.warning(f"SHAP explainer not found at {explainer_path}. Explanations disabled.")
    else:
        logger.info(f"Loaded SHAP explainer from {explainer_path}")

    # Load XGBoost (optional - for comparison)
    xgb_path = MODELS_DIR / "xgboost.pkl"
    if xgb_path.exists():
        with open(xgb_path, "rb") as f:
            model_xgb = pickle.load(f)
        logger.info("Loaded XGBoost model for comparison")
    else:
        logger.warning("XGBoost model not found. Comparison features disabled.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    load_artifacts()
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn based on telecom features",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerData(BaseModel):
    """Input schema for single prediction."""

    tenure: int = Field(..., ge=0, le=100, description="Number of months as a customer")
    MonthlyCharges: float = Field(..., ge=0, le=200, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, le=10000, description="Total charges")
    gender: str = Field(..., pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")
    PhoneService: str = Field(..., pattern="^(Yes|No)$")
    MultipleLines: str = Field(..., pattern="^(Yes|No|No phone service)$")
    InternetService: str = Field(..., pattern="^(DSL|Fiber optic|No)$")
    OnlineSecurity: str = Field(..., pattern="^(Yes|No|No internet service)$")
    OnlineBackup: str = Field(..., pattern="^(Yes|No|No internet service)$")
    DeviceProtection: str = Field(..., pattern="^(Yes|No|No internet service)$")
    TechSupport: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingTV: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingMovies: str = Field(..., pattern="^(Yes|No|No internet service)$")
    Contract: str = Field(..., pattern="^(Month-to-month|One year|Two year)$")
    PaperlessBilling: str = Field(..., pattern="^(Yes|No)$")
    PaymentMethod: str = Field(
        ...,
        pattern=r"^(Electronic check|Mailed check|Bank transfer \(automatic\)|Credit card \(automatic\))$",
    )


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""

    customers: List[CustomerData]


class PredictionResponse(BaseModel):
    """Output schema for predictions."""

    churn_probability: float
    churn_prediction: int
    churn_label: str


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""

    predictions: List[PredictionResponse]


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "explainer_loaded": explainer is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(customer: CustomerData):
    """Predict churn for a single customer."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")

    try:
        input_df = preprocess_for_inference(customer.dict(), expected_columns, scaler)
        prob = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]

        return PredictionResponse(
            churn_probability=round(float(prob), 4),
            churn_prediction=int(pred),
            churn_label="Yes" if pred == 1 else "No",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/explain", tags=["Predictions"])
async def predict_explain(customer: CustomerData):
    """Explain a churn prediction with SHAP values."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    if explainer is None:
        raise HTTPException(status_code=500, detail="SHAP explainer not loaded")

    try:
        input_df = preprocess_for_inference(customer.dict(), expected_columns, scaler)
        explanation = explain_prediction(explainer, input_df)

        # Also get the prediction
        prob = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]

        return {
            "prediction": {
                "churn_probability": round(float(prob), 4),
                "churn_prediction": int(pred),
                "churn_label": "Yes" if pred == 1 else "No",
            },
            "explanation": explanation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/compare", tags=["Predictions"])
async def compare_models(customer: CustomerData):
    """Compare predictions between RandomForest and XGBoost models."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="RandomForest model not loaded")
    if model_xgb is None:
        raise HTTPException(status_code=500, detail="XGBoost model not loaded")

    try:
        input_df = preprocess_for_inference(customer.dict(), expected_columns, scaler)

        # RandomForest prediction
        rf_prob = model.predict_proba(input_df)[0, 1]
        rf_pred = model.predict(input_df)[0]

        # XGBoost prediction
        xgb_prob = model_xgb.predict_proba(input_df)[0, 1]
        xgb_pred = model_xgb.predict(input_df)[0]

        # Determine agreement
        agree = rf_pred == xgb_pred
        confidence_diff = abs(rf_prob - xgb_prob)

        return {
            "random_forest": {
                "churn_probability": round(float(rf_prob), 4),
                "churn_prediction": int(rf_pred),
                "churn_label": "Yes" if rf_pred == 1 else "No",
            },
            "xgboost": {
                "churn_probability": round(float(xgb_prob), 4),
                "churn_prediction": int(xgb_pred),
                "churn_label": "Yes" if xgb_pred == 1 else "No",
            },
            "comparison": {
                "models_agree": agree,
                "confidence_difference": round(float(confidence_diff), 4),
                "average_probability": round(float((rf_prob + xgb_prob) / 2), 4),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(batch: BatchPredictionInput):
    """Predict churn for multiple customers."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")

    predictions = []
    try:
        for customer in batch.customers:
            input_df = preprocess_for_inference(customer.dict(), expected_columns, scaler)
            prob = model.predict_proba(input_df)[0, 1]
            pred = model.predict(input_df)[0]

            predictions.append(
                PredictionResponse(
                    churn_probability=round(float(prob), 4),
                    churn_prediction=int(pred),
                    churn_label="Yes" if pred == 1 else "No",
                )
            )

        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch/csv", tags=["Predictions"])
async def predict_batch_csv(file: UploadFile = File(...)):
    """Predict churn for multiple customers from CSV file."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")

    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Validate required columns
        required_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_cols)}",
            )

        # Use shared preprocessing
        df_processed = preprocess_batch_csv(df, expected_columns, scaler)

        # Predict for all rows
        probs = model.predict_proba(df_processed)[:, 1]
        preds = model.predict(df_processed)

        # Build results
        results = []
        for prob, pred in zip(probs, preds):
            results.append({
                "churn_probability": round(float(prob), 4),
                "churn_prediction": int(pred),
                "churn_label": "Yes" if pred == 1 else "No",
                "risk_level": (
                    "High" if prob > 0.5 else "Medium" if prob > 0.3 else "Low"
                ),
            })

        # Add results to original dataframe
        results_df = pd.DataFrame(results)
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        # Convert to CSV
        output = io.StringIO()
        output_df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch/dataset", tags=["Predictions"])
async def predict_batch_dataset():
    """Predict churn for all customers in the built-in dataset."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")

    try:
        dataset_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Built-in dataset not found")

        df = pd.read_csv(dataset_path)
        df_processed = preprocess_batch_csv(df, expected_columns, scaler)

        # Predict
        probs = model.predict_proba(df_processed)[:, 1]
        preds = model.predict(df_processed)

        # Build results
        results = []
        for prob, pred in zip(probs, preds):
            results.append({
                "churn_probability": round(float(prob), 4),
                "churn_prediction": int(pred),
                "churn_label": "Yes" if pred == 1 else "No",
                "risk_level": (
                    "High" if prob > 0.5 else "Medium" if prob > 0.3 else "Low"
                ),
            })

        results_df = pd.DataFrame(results)
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        # Return as JSON
        records = output_df.to_dict(orient="records")
        return {"total": len(records), "predictions": records}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/customer/{customer_id}", tags=["Predictions"])
async def get_customer_prediction(customer_id: str):
    """Get prediction details for a specific customer from the dataset."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")

    try:
        dataset_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Built-in dataset not found")

        df = pd.read_csv(dataset_path)
        customer_df = df[df["customerID"] == customer_id]

        if customer_df.empty:
            raise HTTPException(
                status_code=404, detail=f"Customer {customer_id} not found"
            )

        df_processed = preprocess_batch_csv(customer_df, expected_columns, scaler)

        # Predict
        prob = model.predict_proba(df_processed)[0, 1]
        pred = model.predict(df_processed)[0]

        # Build result
        record = customer_df.iloc[0].to_dict()
        record["churn_probability"] = round(float(prob), 4)
        record["churn_prediction"] = int(pred)
        record["churn_label"] = "Yes" if pred == 1 else "No"
        record["risk_level"] = (
            "High" if prob > 0.5 else "Medium" if prob > 0.3 else "Low"
        )

        return record

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/feature-importance", tags=["Analytics"])
async def get_feature_importance():
    """Get global feature importance from the trained model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        importances = model.feature_importances_
        feature_names = list(model.feature_names_in_)

        importance_data = [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(feature_names, importances)
        ]
        importance_data.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "model": "random_forest",
            "features": importance_data[:15],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/churn-profile", tags=["Analytics"])
async def get_churn_profile():
    """Get churn rate profiles by different dimensions."""
    try:
        dataset_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Built-in dataset not found")

        df = pd.read_csv(dataset_path)
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        def churn_rate(group):
            return {
                "group": str(group.name),
                "total": len(group),
                "churned": int(group["Churn"].sum()),
                "churn_rate": round(float(group["Churn"].mean()) * 100, 1),
            }

        by_contract = (
            df.groupby("Contract").apply(churn_rate, include_groups=False).tolist()
        )

        df["tenure_bucket"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 100],
            labels=["0-12 months", "12-24 months", "24-48 months", "48+ months"],
        )
        by_tenure = (
            df.groupby("tenure_bucket", observed=False)
            .apply(churn_rate, include_groups=False)
            .tolist()
        )

        by_payment = (
            df.groupby("PaymentMethod").apply(churn_rate, include_groups=False).tolist()
        )

        by_internet = (
            df.groupby("InternetService")
            .apply(churn_rate, include_groups=False)
            .tolist()
        )

        return {
            "by_contract": by_contract,
            "by_tenure_bucket": by_tenure,
            "by_payment_method": by_payment,
            "by_internet_service": by_internet,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/check", tags=["Alerts"])
async def check_alerts(threshold: float = 0.5):
    """Check if any customers have high churn risk above threshold."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")

    try:
        dataset_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        df = pd.read_csv(dataset_path)
        df_processed = preprocess_batch_csv(df, expected_columns, scaler)

        # Get predictions
        probs = model.predict_proba(df_processed)[:, 1]

        # Filter high-risk customers
        high_risk_mask = probs >= threshold
        high_risk_df = df[high_risk_mask].copy()
        high_risk_df["churn_probability"] = probs[high_risk_mask]

        # Get top 10 highest risk
        top_risk = high_risk_df.nlargest(10, "churn_probability")[
            ["customerID", "churn_probability", "Contract", "tenure", "MonthlyCharges"]
        ].to_dict("records")

        return {
            "threshold": threshold,
            "total_customers": len(df),
            "high_risk_count": int(high_risk_mask.sum()),
            "high_risk_percentage": round(float(high_risk_mask.mean()) * 100, 2),
            "top_risk_customers": top_risk,
            "alert_triggered": high_risk_mask.any(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/drift", tags=["Monitoring"])
async def check_data_drift():
    """Check for potential data drift by comparing current data statistics
    against baseline (training data) statistics.
    """
    try:
        dataset_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        df = pd.read_csv(dataset_path)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        current_stats = {}

        for col in numeric_cols:
            current_stats[col] = {
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
            }

        baseline_stats = {
            "tenure": {"mean": 32.37, "std": 24.56, "min": 0.0, "max": 72.0},
            "MonthlyCharges": {"mean": 64.76, "std": 30.09, "min": 18.25, "max": 118.75},
            "TotalCharges": {"mean": 2283.30, "std": 2266.77, "min": 18.80, "max": 8684.80},
        }

        drift_metrics = {}
        for col in numeric_cols:
            mean_diff = abs(current_stats[col]["mean"] - baseline_stats[col]["mean"])
            drift_score = mean_diff / baseline_stats[col]["std"]

            drift_metrics[col] = {
                "current_mean": current_stats[col]["mean"],
                "baseline_mean": baseline_stats[col]["mean"],
                "mean_difference": round(mean_diff, 4),
                "drift_score": round(drift_score, 4),
                "drift_detected": drift_score > 0.5,
            }

        any_drift = any(m["drift_detected"] for m in drift_metrics.values())

        return {
            "drift_detected": any_drift,
            "drift_metrics": drift_metrics,
            "dataset_size": len(df),
            "check_timestamp": pd.Timestamp.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }
