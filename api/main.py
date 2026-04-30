"""FastAPI backend for churn prediction."""
import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load model and scaler
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn based on telecom features",
    version="1.0.0",
)

# Load artifacts on startup
model = None
scaler = None


def load_artifacts():
    """Load trained model and scaler."""
    global model, scaler
    model_path = MODELS_DIR / "random_forest.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"

    if not model_path.exists():
        print(f"WARNING: Model not found at {model_path}. Predictions will not work until training is run.")
        return
    if not scaler_path.exists():
        print(f"WARNING: Scaler not found at {scaler_path}. Predictions will not work until training is run.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)


@app.on_event("startup")
async def startup_event():
    load_artifacts()


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
        pattern="^(Electronic check|Mailed check|Bank transfer \(automatic\)|Credit card \(automatic\))$"
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


def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data for prediction."""
    df = pd.DataFrame([data])

    # Encode binary
    binary_mappings = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
    }
    for col, mapping in binary_mappings.items():
        df[col] = df[col].map(mapping)

    # One-hot encode categoricals
    cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align columns with training data
    expected_cols = set(model.feature_names_in_)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[list(expected_cols)]

    # Scale numeric
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(customer: CustomerData):
    """Predict churn for a single customer."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        input_df = preprocess_input(customer.dict())
        prob = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]

        return PredictionResponse(
            churn_probability=round(float(prob), 4),
            churn_prediction=int(pred),
            churn_label="Yes" if pred == 1 else "No",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(batch: BatchPredictionInput):
    """Predict churn for multiple customers."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    predictions = []
    try:
        for customer in batch.customers:
            input_df = preprocess_input(customer.dict())
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


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }
